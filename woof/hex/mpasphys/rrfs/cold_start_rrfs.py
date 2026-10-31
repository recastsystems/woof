"""RRFS-MPAS cold start: initial conditions (and lateral boundaries) through the init door.

RRFS itself starts from GFS with NOAA GSL's init_atmosphere fork (RRFSx/MPAS-Model-rrfs
@1d6e370e): case 7 for the initial state, case 9 hourly for the boundaries, from a static
made with the fork's land-use, soil and LAI sources and the GSL drag statistics file
ugwp_oro_data.nc. This module turns a producer row of ``data/profiles/init-producers.v1.json``
plus the door's paths into the job sequence of the Rust producer ``rw_mpas_init_v831``
(hex-init-job/v1) and runs it. No weather array is read or written here: the producer
decodes, interpolates and writes; this module only resolves the recipe from the table.

What runs (all Rust, all cores):

* ``static`` (only when the given static lacks the fields the suite reads, and a WPS
  geography root is given): the fork static (mpas_init_atm_static.F:110-1424,1957-2120);
* ``prepare``: horizontal fields, vertical grid (v8.3.1 L60, cases.F:4448-4905) and the
  init carrier;
* ``case7``: the initial state (cases.F:4043-7742) with RUC's nine soil nodes
  (init_soil_properties_ruc), TEMPO aerosols source-first and first-guess hydrometeors;
* ``case9`` per boundary time (cases.F:7541-8986) on a carrier made from the case-7 init;
* the GSL orographic statistics (rw_mpas_gsl_oro) when the geography holds topo_ugwp.

The producer's numbers were proven against the fork's own strict native init_atmosphere,
every stored value, on the module replay recorded in docs/rrfs-init.md.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Sequence

PRODUCERS_PATH = "data/profiles/init-producers.v1.json"
SCHEMA = "gpuwm-hex.init-producers/v1"


class RrfsInitError(ValueError):
    """The RRFS producer could not run: what breaks, and what fixes it."""


def _profile(name: str) -> Any:
    return json.loads(resources.files("woof.hex").joinpath("data/profiles/" + name).read_text(encoding="utf-8"))


def load_producers() -> Mapping[str, Any]:
    table = json.loads(resources.files("woof.hex").joinpath(PRODUCERS_PATH).read_text(encoding="utf-8"))
    if table.get("schema") != SCHEMA:
        raise RrfsInitError(f"{PRODUCERS_PATH} schema {table.get('schema')!r} is not {SCHEMA!r}")
    return table


def producer_for_suite(suite: str | None, *, pin_set: str | None = None) -> tuple[str, Mapping[str, Any]] | None:
    """The producer row that initializes ``suite``, or None when the door's own init serves it."""

    if not suite:
        return None
    from ...physics_suites import load_table, route_default_for
    producers = load_producers()["producers"]
    routes = [row for row in load_table().values() if isinstance(row, dict)
              and suite in row.get("suite_routes", {})
              and (route_default_for(row, suite) if pin_set is None else row.get("pin_set") == pin_set)]
    # An explicit pin selects its route before default filtering.
    for route in routes[:1]:
        if "init_producer" in route:
            key = route["init_producer"]
            if key not in producers:
                raise RrfsInitError(f"route names missing init producer {key!r}")
            return key, producers[key]
    for key, row in producers.items():
        route = load_table().get(row["physics_table_key"], {})
        if pin_set is not None and route.get("pin_set") != pin_set:
            continue
        if pin_set is None and not route_default_for(route, suite):
            continue
        if suite in row["suite_routes"]:
            return key, row
    if pin_set is not None:
        raise RrfsInitError(f"no init producer for suite {suite!r} and pin set {pin_set!r}")
    return None


def producer_suites() -> set[str]:
    return {suite for row in load_producers()["producers"].values() for suite in row["suite_routes"]}


# ---------------------------------------------------------------------------
# engines
# ---------------------------------------------------------------------------
def resolve_worker(row: Mapping[str, Any], explicit: str | Path | None = None) -> Path:
    """The producer's Rust worker through the shared engine ladder, ABI-checked.

    A worker whose ``--abi`` answer is not the row's would read the jobs under a
    different contract and write an init the forecast cannot trust, so it is refused.
    """

    from ...engines import INIT_RRFS, EngineRefusal, resolve

    try:
        worker = resolve(INIT_RRFS, explicit)
    except EngineRefusal as error:
        raise RrfsInitError(str(error)) from None
    probe = subprocess.run([str(worker), "--abi"], capture_output=True, text=True)
    if probe.returncode or probe.stdout.strip() != row["worker"]["abi"]:
        raise RrfsInitError(
            f"{worker} answers --abi with {probe.stdout.strip()!r}, not this producer's "
            f"{row['worker']['abi']!r}; its jobs would be read under another contract. "
            "Stage the rw_mpas_init_v831 of this release (woof fetch-bridges)")
    return worker


def resolve_gsl_oro(explicit: str | Path | None = None) -> Path:
    from ...engines import GSL_ORO, EngineRefusal, resolve

    try:
        return resolve(GSL_ORO, explicit)
    except EngineRefusal as error:
        raise RrfsInitError(str(error)) from None


# ---------------------------------------------------------------------------
# recipe -> jobs
# ---------------------------------------------------------------------------
def declarations(row: Mapping[str, Any], start_time: str, part: str) -> dict[str, Any]:
    document = _profile(row["declarations"])
    if document.get("source_pin") != row["source_pin"]:
        raise RrfsInitError(f"{row['declarations']} is for {document.get('source_pin')}, not {row['source_pin']}")
    section = document[part]
    variables = json.loads(json.dumps(section["variables"]).replace("@START_TIME@", start_time))
    return dict(section, variables=variables)


def specified_levels_text(row: Mapping[str, Any]) -> str:
    """The specified zeta levels as the fork's formatted reader takes them (one per line)."""

    spec = _profile(row["vertical"]["specified_levels"])
    levels = spec["specified_interfaces_m"]
    if len(levels) != row["vertical"]["levels"] + 1 or float(levels[-1]) != float(row["vertical"]["top"]):
        raise RrfsInitError(f"{row['vertical']['specified_levels']} does not hold {row['vertical']['levels']} "
                            f"layers up to {row['vertical']['top']} m")
    return "".join(f"{value!r}\n" if value else "0\n" for value in levels)


def static_missing_fields(static: Path, row: Mapping[str, Any]) -> list[str]:
    from netCDF4 import Dataset

    with Dataset(static) as dataset:
        present = set(dataset.variables)
        soil_cats = len(dataset.dimensions["nSoilCat"]) if "nSoilCat" in dataset.dimensions else None
    missing = [name for name in row["static"]["fields_read_by_the_suite"] if name not in present]
    if soil_cats not in (None, 16) and "soilf" in present:
        missing.append(f"soilf over BNU's 16 categories (this static has {soil_cats})")
    return missing


def mesh_is_regional(path: Path) -> bool:
    """A limited-area mesh marks its boundary cells (bdyMaskCell > 0); a global one marks none."""

    from netCDF4 import Dataset

    with Dataset(path) as dataset:
        if "bdyMaskCell" not in dataset.variables:
            return False
        return bool(dataset.variables["bdyMaskCell"][:].max() > 0)


#: The init door's flag that picks the deep-soil temperature the fork static is built with.
DEEP_SOIL_FLAG = "--deep-soil-temperature"


def deep_soil_choice(row: Mapping[str, Any], choice: str | None = None) -> tuple[str | None, dict[str, Any]]:
    """The deep-soil temperature dataset a fork static built here reads: the row's default (ERA5), or ``choice``.

    THE DECISION THIS CARRIES (the project owner, 2026-10-31): NOAA's recipe reads ISLSCP-I's soiltemp_1deg, whose licence is
    unclear and which woof holds out of the default geography; a static built here from world-wide geography takes
    the permissive ERA5 field instead, and the receipt says so.  NOAA's own static (the conus3km preset) is not
    rebuilt, so that preset keeps NOAA's field exactly.  ISLSCP-I stays selectable by name, with its licence note.
    """

    recipe = row["static"]
    choices = recipe.get("soil_temperature_choices") or {}
    if not choices:
        if choice is not None:
            raise RrfsInitError(f"{DEEP_SOIL_FLAG} {choice}: this producer row names no deep-soil choices")
        return None, {"dataset": recipe["geog"]["soil_temperature"]}
    name = choice or recipe.get("soil_temperature_default")
    if name not in choices:
        raise RrfsInitError(f"{DEEP_SOIL_FLAG} {name!r} is not one of {', '.join(sorted(choices))}")
    return name, dict(choices[name])


def static_job(row: Mapping[str, Any], *, grid: Path, geog: Path, out: Path, valid_time: str,
               tile_workers: int | None = None, deep_soil: str | None = None) -> dict[str, Any]:
    recipe = row["static"]
    _, chosen = deep_soil_choice(row, deep_soil)
    datasets = dict(recipe["geog"], soil_temperature=chosen["dataset"])
    missing = [name for name in datasets.values() if not (geog / name).is_dir()]
    if missing:
        raise RrfsInitError(
            f"the WPS geography at {geog} lacks {', '.join(missing)}, which the fork static reads "
            "(MODIS 15 s land use, BNU soils, MODIS LAI, the deep-soil temperature); stage them with woof "
            "fetch-geog (its default set builds soiltemp_era5_1deg from ERA5) or point --geog at a root that has them")
    return {"stage": "static", "config": {
        "nvegopt": recipe["nvegopt"], "gwd_topo_shift": recipe["gwd_topo_shift"],
        "gwd_cell_scaling": recipe["gwd_cell_scaling"], "grid_path": str(grid), "out_path": str(out),
        "geog": {key: str(geog / name) for key, name in datasets.items()},
        "stages": {"vegetation_ncep": False, "albedo_ncep": False, "snow_ncep": False,
                   "noahmp": recipe["noahmp"], "gwd": recipe["gwd"],
                   "landuse_label": recipe["landuse_label"], "native_init": True},
        "supersample": recipe["supersample"], "supersample_landuse": recipe["supersample_landuse"],
        "supersample_30s": recipe["supersample_30s"], "host_memory_limit_bytes": None,
        "tile_workers": tile_workers, "receipt_path": str(out.with_name(out.name + ".build.json")),
        "provenance": f"woof hex init, RRFS producer {row['source_pin'][:8]} static",
        "valid_time": valid_time, "nominal_dx_m": None, "clobber": True,
        "coordinates": recipe["coordinates"]}}


def _vertical(row: Mapping[str, Any], *, regional: bool, boundary_met: Path) -> dict[str, Any]:
    # A global mesh hands the fork its levels file (config_specified_zeta_levels); a limited
    # area hands the same interfaces inline with the regional halo on. Both forms are the
    # ones the module replay proved (canonical global rows and the CONUS cut rows).
    v = row["vertical"]
    specified = None
    scheme = v["scheme"]
    if regional:
        scheme = "specified"
        specified = [float(x) for x in _profile(v["specified_levels"])["specified_interfaces_m"]]
    result = {"levels": v["levels"], "top": v["top"], "scheme": scheme, "specified": specified,
            "projection": v["projection"], "terrain_passes": v["terrain_passes"], "smooth": v["smooth"],
            "surface_passes": v["surface_passes"], "minimum_fraction": v["minimum_fraction"],
            "hybrid": v["hybrid"], "transition": v["transition"], "damping": v["damping"],
            "damping_start": v["damping_start"], "regional_halo": bool(regional), "order": v["order"],
            "boundary": {"met_path": str(boundary_met), "lat": [], "lon": [], "mask": []}}
    if "coordinate_recipe" in v:
        result["coordinate_recipe"] = v["coordinate_recipe"]
    return result


def case7_jobs(row: Mapping[str, Any], *, static: Path, met: Path, start_time: str, nfglevels: int,
               nfgsoillevels: int, artifact: Path, out: Path, levels_file: Path, aerosols: Path,
               regional: bool = False, sfc_met: Path | None = None, seaice_met: Path | None = None,
               version: str = "") -> list[dict[str, Any]]:
    """``prepare`` then ``case7``, exactly as the fork's hrrrv5 init_atmosphere namelist sets them."""

    decl = declarations(row, start_time, "prepare")
    prepare = {"stage": "prepare", "config": {
        "static_path": str(static), "output_path": str(artifact),
        "soil_levels": row["soil"]["n_soil_layers"], "met_interp": True,
        "vertical": _vertical(row, regional=regional, boundary_met=met),
        "variables": decl["variables"], "attributes": decl["attributes"]}}
    if "omit_fields" in decl:
        prepare["config"]["omit_fields"] = decl["omit_fields"]
    if not regional:
        prepare["config"]["specified_levels_path"] = str(levels_file)
    c = row["case7"]
    case7 = {"stage": "case7", "config": {
        "tempo_rap": c["tempo_rap"], "smoke": c["smoke"], "dust": c["dust"],
        "n_soil_layers": row["soil"]["n_soil_layers"], "nvegopt": c["nvegopt"],
        "met_path": str(met), "seaice_met_path": str(seaice_met) if seaice_met else None,
        "omld_met_path": None, "sfc_met_path": str(sfc_met) if sfc_met else None,
        "gocart_path": str(aerosols), "gocart_levels": c["gocart_levels"],
        "static_path": str(static), "capsule_path": str(artifact), "reference_path": str(artifact),
        "out_path": str(out), "start_time": start_time, "n_fg_levels": int(nfglevels),
        "n_fg_soil_levels": int(nfgsoillevels), "extrap_airtemp": c["extrap_airtemp"],
        "use_spechumd": c["use_spechumd"], "theta_adv_order": c["theta_adv_order"],
        "coef_3rd_order": c["coef_3rd_order"], "virtual_factor": c["virtual_factor"],
        "deep_moisture": c["deep_moisture"], "landuse_table": c["landuse_table"],
        "frac_seaice": c["frac_seaice"], "tsk_seaice_threshold": c["tsk_seaice_threshold"],
        "oned_underflow": c["oned_underflow"],
        "provenance": f"woof hex init, RRFS producer {row['source_pin'][:8]}",
        "lineage": {**row["lineage"], "version": version or "unknown",
                    "git_version": f"gpuwm-hex-{version}" if version else "woof hex"}}}
    if "seaice_cleanup" in c:
        case7["config"]["seaice_cleanup"] = c["seaice_cleanup"]
    return [prepare, case7]


def case9_jobs(row: Mapping[str, Any], *, init: Path, start_time: str, intervals: Sequence[tuple[str, Path]],
               nfglevels: int, out_dir: Path, carrier: Path, aerosols: Path) -> list[dict[str, Any]]:
    """The grid carrier (from the case-7 init) and one ``case9`` job over every boundary time."""

    decl = declarations(row, start_time, "artifact")
    artifact = {"stage": "artifact", "config": {
        "template": str(init), "output": str(carrier), "dimensions": decl["dimensions"],
        "variables": decl["variables"], "attributes": decl["attributes"]}}
    c, lbc = row["case7"], row["lbc"]
    times = [t for t, _ in intervals]
    case9 = {"stage": "case9", "config": {
        "hydrometeors_rrfs": lbc["hydrometeors_rrfs"], "hydrometeors_gfs": lbc["hydrometeors_gfs"],
        "tempo_rap": c["tempo_rap"], "aerosolaware": lbc["aerosolaware"], "smoke": c["smoke"],
        "dust": c["dust"], "gocart_path": str(aerosols), "gocart_levels": c["gocart_levels"],
        "grid_path": str(carrier), "out_dir": str(out_dir), "start_time": times[0], "stop_time": times[-1],
        "intervals": [{"valid_time": t, "met_path": str(p)} for t, p in intervals],
        "n_fg_levels": int(nfglevels), "extrap_airtemp": c["extrap_airtemp"],
        "use_spechumd": c["use_spechumd"], "theta_adv_order": c["theta_adv_order"],
        "coef_3rd_order": c["coef_3rd_order"], "fg_interval_seconds": lbc["fg_interval_seconds"],
        "oned_underflow": c["oned_underflow"], "attr_overrides": {},
        "provenance": f"woof hex init, RRFS producer {row['source_pin'][:8]} boundaries"}}
    return [artifact, case9]


def run_jobs(worker: Path, jobs: Sequence[Mapping[str, Any]], logs: Path) -> list[dict[str, Any]]:
    """Run each job through the worker (its stdin is the job); every stage is Rust on all cores."""

    receipts = []
    logs.mkdir(parents=True, exist_ok=True)
    for index, job in enumerate(jobs):
        started = time.monotonic()
        log = logs / f"{index:02d}-{job['stage']}.log"
        with log.open("wb") as stream:
            completed = subprocess.run([str(worker)], input=json.dumps(job).encode(),
                                       stdout=subprocess.PIPE, stderr=stream)
        if completed.returncode:
            raise RrfsInitError(f"RRFS producer stage {job['stage']} failed (rc {completed.returncode}); "
                                f"its log is {log}")
        try:
            result = json.loads(completed.stdout or b"null")
        except json.JSONDecodeError:
            result = {"stdout": completed.stdout.decode("utf-8", "replace")[-4000:]}
        receipts.append({"stage": job["stage"], "seconds": round(time.monotonic() - started, 3),
                         "log": str(log), "result": result})
    return receipts


def resolve_aerosol_table(explicit: Path | None) -> Path:
    """The monthly TEMPO aerosol climatology the fork reads when the first guess has none."""

    from woof.ingest.wif_climatology import resolve_wif_climatology

    resolution = resolve_wif_climatology(explicit)
    if not resolution.resolved:
        from woof.table_assets import stage_wif_dataset
        from woof.ingest.wif_dataset import resolve_wif_data_root, WIF_DATASET_FILE

        if stage_wif_dataset() != 0:
            raise RrfsInitError("the RRFS init needs QNWFA_QNIFA_SIGMA_MONTHLY.dat for TEMPO's nwfa/nifa when "
                                "the first guess carries none, and staging it failed; pass --wif-climatology")
        resolution = resolve_wif_climatology(resolve_wif_data_root() / WIF_DATASET_FILE)
    if not resolution.resolved:
        raise RrfsInitError("QNWFA_QNIFA_SIGMA_MONTHLY.dat did not resolve; pass --wif-climatology")
    return Path(resolution.path)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def run_rrfs_init(arguments: Any, producer: tuple[str, Mapping[str, Any]]) -> int:
    """``woof hex init --physics-suite <RRFS suite>``: the fork's case-7 init (and case-9 boundaries)."""

    from ... import __version__

    key, row = producer
    started = time.monotonic()
    for flag, value in (("--met", arguments.met), ("--static", arguments.static), ("--out", arguments.out),
                        ("--start-time", arguments.start_time), ("--nfglevels", arguments.nfglevels),
                        ("--nfgsoillevels", arguments.nfgsoillevels)):
        if value is None:
            raise RrfsInitError(f"{flag} was not given; the RRFS producer cannot infer it")
    if getattr(arguments, "land_surface", None) not in (None, row["soil"]["land_surface"]):
        raise RrfsInitError(f"--physics-suite {arguments.physics_suite} runs {row['soil']['land_surface']} "
                            f"on {row['soil']['n_soil_layers']} soil levels; drop --land-surface "
                            f"{arguments.land_surface}")
    out = Path(arguments.out).expanduser().absolute()
    if not out.parent.is_dir():
        raise RrfsInitError(f"output directory {out.parent} does not exist; create it first")
    from ...init_door import resolve_met_source

    met = resolve_met_source(Path(arguments.met).expanduser()).resolve()
    static = Path(arguments.static).expanduser().absolute()
    grid = Path(arguments.grid).expanduser().absolute() if arguments.grid else None
    from ...hrrr_intermediate import uncovered_cells_refusal

    uncovered = uncovered_cells_refusal(met, grid or static)
    if uncovered:
        raise RrfsInitError(uncovered)
    geog = Path(arguments.geog).expanduser().absolute() if getattr(arguments, "geog", None) else None
    work = out.with_name(out.name + ".rrfs-work")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    worker = resolve_worker(row, getattr(arguments, "rrfs_worker", None))
    jobs: list[dict[str, Any]] = []
    missing = static_missing_fields(static, row)
    if missing:
        if geog is None or grid is None:
            raise RrfsInitError(
                f"{static} lacks {', '.join(missing)}, which --physics-suite {arguments.physics_suite} reads "
                "(RUC: landusef/soilf, driver_lsm.F:471-514; the surface update: lai12m, update_surface.F:101-140). "
                "Pass --grid and --geog (a WPS geography root with MODIS 15 s land use, BNU soils and MODIS LAI) "
                "and the init builds the fork static first, or pass a static the fork's init_atmosphere made")
        rrfs_static = out.with_name(out.name + ".rrfs-static.nc")
        choice, chosen = deep_soil_choice(row, getattr(arguments, "deep_soil_temperature", None))
        if chosen.get("licence_hold"):
            import sys
            print(f"init: deep-soil temperature {chosen['dataset']} selected by name: {chosen['licence_hold']}",
                  file=sys.stderr)
        deep_soil = {"choice": choice, "built_here": True, **chosen}
        jobs.append(static_job(row, grid=grid, geog=geog, out=rrfs_static, valid_time=arguments.start_time,
                               deep_soil=choice))
        static = rrfs_static
    else:
        deep_soil = {"built_here": False,
                     "about": f"the given static {static.name} carries its own deep-soil temperature (soiltemp); "
                              "no fork static was built, so none was chosen here"}
    levels = work / "ZETA_LEVELS.txt"
    levels.write_text(specified_levels_text(row))
    aerosols = resolve_aerosol_table(getattr(arguments, "wif_climatology", None))
    regional = mesh_is_regional(static if not missing else (grid or static))
    jobs += case7_jobs(row, static=static, met=met, start_time=arguments.start_time,
                       nfglevels=arguments.nfglevels, nfgsoillevels=arguments.nfgsoillevels,
                       artifact=work / "artifact.nc", out=out, levels_file=levels, aerosols=aerosols,
                       regional=regional, sfc_met=getattr(arguments, "sfc_met", None),
                       seaice_met=getattr(arguments, "seaice_met", None), version=__version__)
    receipts = run_jobs(worker, jobs, work / "logs")
    _refuse_cold_land_soil(out, row, int(arguments.nfgsoillevels))
    lbc_files: list[str] = []
    lbc_met = list(getattr(arguments, "lbc_met", None) or [])
    if lbc_met:
        intervals = []
        for spec in lbc_met:
            when, _, path = str(spec).partition("=")
            if not path:
                raise RrfsInitError(f"--lbc-met {spec!r} is not TIME=MET_PATH")
            intervals.append((when, resolve_met_source(Path(path).expanduser()).resolve()))
        lbc_dir = Path(arguments.lbc_dir).expanduser().absolute() if getattr(arguments, "lbc_dir", None) \
            else out.parent
        lbc_dir.mkdir(parents=True, exist_ok=True)
        receipts += run_jobs(worker, case9_jobs(row, init=out, start_time=arguments.start_time,
                                                intervals=intervals, nfglevels=arguments.nfglevels,
                                                out_dir=lbc_dir, carrier=work / "carrier.nc",
                                                aerosols=aerosols), work / "logs-lbc")
        lbc_files = sorted(str(p) for p in lbc_dir.glob("lbc.*.nc"))
    oro = _gsl_orography(row, grid=grid, geog=geog, out=out.parent / row["gwd"]["file"],
                         choice=getattr(arguments, "ugwp_oro", None) or row["gwd"]["default"], work=work)
    payload = {
        "schema": "gpuwm-hex.init-provenance/rrfs-v1",
        "tool": "woof hex init",
        "door_version": __version__,
        "physics_suite": arguments.physics_suite,
        "producer": {"key": key, "source_pin": row["source_pin"], "worker": str(worker),
                     "worker_sha256": _sha256(worker), "abi": row["worker"]["abi"]},
        "recipe": {name: row[name] for name in ("soil", "vertical", "case7", "aerosols", "hydrometeors", "lbc")},
        "inputs": {"met": str(met), "met_sha256": _sha256(met), "static": str(static),
                   "aerosol_climatology": str(aerosols), "aerosol_climatology_sha256": _sha256(aerosols)},
        "deep_soil_temperature": deep_soil,
        "stages": receipts,
        "output": {"path": str(out), "bytes": out.stat().st_size, "sha256": _sha256(out)},
        "lbc": lbc_files,
        "ugwp_oro": oro,
        "seconds": round(time.monotonic() - started, 3),
    }
    receipt = Path(arguments.receipt).expanduser() if getattr(arguments, "receipt", None) \
        else out.with_name(out.name + ".provenance.json")
    receipt.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    # The carriers are intermediate files the producer rebuilds on every run.
    (work / "artifact.nc").unlink(missing_ok=True)
    (work / "carrier.nc").unlink(missing_ok=True)
    return 0


def _refuse_cold_land_soil(out: Path, row: Mapping[str, Any], nfgsoillevels: int) -> None:
    """Refuse (and remove) an init whose land soil the producer left at its cold zeros.

    THE BREAKAGE IT PREVENTS (hrrrv5 from ICON, 2026-10-29): the fork's init_soil_properties_ruc builds RUC's
    nine levels from Noah's four first-guess layers or copies RUC's own nine nodes; ICON's eight soil levels
    take neither branch, and every land cell started at 0 K through all nine levels with rc 0 (HRDPS's one
    slab did the same before its deep-soil donor)."""

    from netCDF4 import Dataset
    import numpy as np

    try:
        with Dataset(out) as dataset:
            if "tslb" not in dataset.variables or "landmask" not in dataset.variables:
                return
            soil = np.asarray(dataset.variables["tslb"][0], dtype=np.float64)
            land = np.asarray(dataset.variables["landmask"][:]).reshape(-1) == 1
    except OSError:
        return  # not a netCDF init (a test worker's stand-in): nothing here to read
    cold = land & ~(soil > 0.0).all(axis=1)
    if cold.any():
        out.unlink(missing_ok=True)
        raise RrfsInitError(
            f"the producer left {int(cold.sum())} of {int(land.sum())} land cells with a soil temperature of 0 K: "
            f"the first guess carries {nfgsoillevels} soil levels, and {row['soil']['recipe']} builds "
            f"{row['soil']['land_surface']}'s column only from Noah's four layers or its own "
            f"{row['soil']['n_soil_layers']} nodes. The init was removed; give the source row those soil layers "
            "(a deep-soil donor, or its layers interpolated to Noah's four)")


def run_rrfs_lbc(producer: tuple[str, Mapping[str, Any]], *, init: Path, intervals: Sequence[tuple[str, Path]],
                 nfglevels: int, out_dir: Path, worker: str | Path | None = None,
                 wif_climatology: Path | None = None) -> dict[str, Any]:
    """The producer's case-9 boundaries alone, on a carrier made from its own case-7 init.

    ``intervals`` are ``(YYYY-MM-DD_HH:MM:SS, WPS intermediate)`` from the start time through the stop
    time.  THE BREAKAGE IT CLOSES: a limited-area run on a producer suite (hrrrv5) got rw_mpas_lbc's
    MPAS-A v8.4.2 case-9 series, on the v8.4.2 fields, for an init made on the fork's L60 grid with its
    hydrometeor and aerosol rows; the fork's own case 9 writes the boundaries that init expects.  The
    first-guess spacing is the series' own (a source with 6-hourly leads drives with 6-hourly files).
    """

    from datetime import datetime

    key, row = producer
    if len(intervals) < 2:
        raise RrfsInitError("a boundary series needs at least two times (the start and the stop)")
    moments = [datetime.strptime(when, "%Y-%m-%d_%H:%M:%S") for when, _ in intervals]
    gaps = {int((b - a).total_seconds()) for a, b in zip(moments, moments[1:])}
    if len(gaps) != 1 or gaps.pop() <= 0:
        raise RrfsInitError("the boundary intermediates are not evenly spaced in time; case 9 reads one "
                            "first-guess interval")
    spacing = int((moments[1] - moments[0]).total_seconds())
    row = dict(row, lbc=dict(row["lbc"], fg_interval_seconds=spacing))
    out_dir = Path(out_dir).expanduser().absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / ".rrfs-lbc-work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    started = time.monotonic()
    resolved = resolve_worker(row, worker)
    aerosols = resolve_aerosol_table(wif_climatology)
    receipts = run_jobs(resolved, case9_jobs(row, init=Path(init), start_time=intervals[0][0],
                                             intervals=list(intervals), nfglevels=int(nfglevels),
                                             out_dir=out_dir, carrier=work / "carrier.nc", aerosols=aerosols),
                        work / "logs")
    (work / "carrier.nc").unlink(missing_ok=True)
    files = sorted(out_dir.glob("lbc.*.nc"))
    if len(files) < len(intervals):
        raise RrfsInitError(f"the RRFS producer's case 9 wrote {len(files)} boundary files for {len(intervals)} "
                            f"times; its logs are in {work / 'logs'}")
    return {"producer": {"key": key, "source_pin": row["source_pin"], "worker": str(resolved),
                         "worker_sha256": _sha256(resolved), "abi": row["worker"]["abi"]},
            "fg_interval_seconds": spacing, "stages": receipts, "seconds": round(time.monotonic() - started, 3),
            "files": [str(path) for path in files]}


def _gsl_orography(row: Mapping[str, Any], *, grid: Path | None, geog: Path | None, out: Path,
                   choice: str, work: Path) -> dict[str, Any]:
    """ugwp_oro_data.nc from the GSL generator (the RRFS default), or the record of who makes it."""

    if choice != "gsl-oro":
        return {"method": choice, "made_here": False,
                "note": "the forecast door makes it with hex's own method (woof.hex.ugwp_oro)"}
    needed = row["gwd"]["geog"]
    if grid is None or geog is None or any(not (geog / name).exists() for name in needed):
        return {"method": "gsl-oro", "made_here": False,
                "note": f"needs --grid and a --geog holding {', '.join(needed)}; the forecast door makes "
                        "it with hex's own method otherwise"}
    engine = resolve_gsl_oro(None)
    log = work / "gsl-oro.log"
    with log.open("wb") as stream:
        completed = subprocess.run([str(engine), "--grid", str(grid), "--geog", str(geog), "--out", str(out),
                                    "--clobber"], stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RrfsInitError(f"rw_mpas_gsl_oro failed (rc {completed.returncode}); its log is {log}")
    return {"method": "gsl-oro", "made_here": True, "path": str(out), "sha256": _sha256(out)}


def select_initial_state_stream(configs: Mapping[str, Any]) -> str:
    """The one initial-state stream the fork core reads.

    MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:139-150 at 1d6e370e: restart when
    config_do_restart, the DA state when config_do_DAcycling, else the cold-start input
    stream this module's init writes.
    """

    configs = {key.casefold(): value for key, value in configs.items()}
    if configs.get("config_do_restart"):
        return "restart"
    if configs.get("config_do_dacycling"):
        return "da_state"
    return "input"
