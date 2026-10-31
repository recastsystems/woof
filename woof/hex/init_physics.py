"""Resolve initialization metadata and invoke the native producers."""
from __future__ import annotations
import json
import subprocess
import shutil
from argparse import Namespace
from pathlib import Path


SUITE_INIT_FLAG = "--aerosol-status"


def engine_has_suite_init(engine) -> bool:
    """Whether this rw_mpas_init carries the suite-aware entries
    (``--aerosol-status``, ``--augment-init``) that woof 2.9.0's init brings.

    An older binary has no such entry and answers either flag with its usage
    ("--met was not given"), exit 1, which reached the user as an unnamed
    ``Command [...] returned non-zero exit status 1``.  Its usage text does
    not list the flag; a suite-aware binary's does.
    """

    probe = subprocess.run([str(engine), "--help"], capture_output=True, text=True)
    return SUITE_INIT_FLAG in (probe.stdout or "") + (probe.stderr or "")


def aerosol_status(engine, init):
    """The engine's ``--aerosol-status`` record for ``init``, or ``None`` when
    the binary predates the suite-aware entries.  Only a failed call probes
    the usage, so a working engine is asked once; a suite-aware engine that
    fails is reported with its own error, as before."""

    completed = subprocess.run([str(engine), "--aerosol-status", str(init)],
                               capture_output=True, text=True)
    if completed.returncode == 0:
        return json.loads(completed.stdout)
    if not engine_has_suite_init(engine):
        return None
    raise subprocess.CalledProcessError(completed.returncode, completed.args,
                                        completed.stdout, completed.stderr)


def _engine_requirement() -> str:
    from .engine_pin import gpuwm_requirement

    return gpuwm_requirement()


def _predates_suite_init(engine, need: str) -> str:
    return (f"{need} needs woof 2.9.0's suite-aware native initializer, and the installed "
            f"rw_mpas_init ({engine}) predates it (its usage lists no {SUITE_INIT_FLAG}). "
            f"Install an engine inside this release's range (pip install '{_engine_requirement()}') or build the init for the suite "
            "with an engine that carries it")


def unknown_suite_refusal(suite, known):
    """The sentence for a --physics-suite no init route serves.

    THE BREAKAGE THIS PREVENTS (hex 0.4.0 recut, 2026-10-16): ``woof hex init --physics-suite <name>`` stopped
    with a bare ``KeyError`` traceback for a suite the door did not know.  Every RRFS-MPAS suite now has its
    producer row (data/profiles/init-producers.v1.json) and every MPAS-A v8.4.2 suite runs from the plain init,
    so only a name no table carries reaches this sentence."""
    from .mpasphys.rrfs.cold_start_rrfs import producer_suites
    return (f"--physics-suite {suite} is not a suite the init door initializes; it knows "
            f"{', '.join(sorted(set(known) | producer_suites()))} and every MPAS-A v8.4.2 suite")


def producer_selection(suite, land_surface, *, pin_set=None):
    """``init --physics-suite`` for a suite whose model initializes with its own producer (RRFS-MPAS).

    The producer row of data/profiles/init-producers.v1.json carries the whole recipe: the fork's case-7 init with
    RUC's nine soil nodes, the v8.3.1 L60 grid, TEMPO aerosols and first-guess hydrometeors, its static and its
    boundaries.  RETIRES (2026-10-28, RRFS native-free item C): "hex does not make RRFS initial conditions in this
    release", which sent every RRFS user to the fork's own Fortran init_atmosphere."""
    from .mpasphys.rrfs.cold_start_rrfs import producer_for_suite
    key, row = producer_for_suite(suite, pin_set=pin_set)
    return {"land_surface": land_surface or row["soil"]["land_surface"], "scalar_names": (),
            "producer": (key, row),
            "physics_initialization": {"suite": suite, "route": row["title"], "source_pin": row["source_pin"]}}


def mpas_v842_selection(suite, land_surface):
    """``init --physics-suite`` for an MPAS-A v8.4.2 suite: the init this door writes with no suite.

    MPAS-A v8.4.2 runs its suites from init_atmosphere's case-7 fields with Noah's four soil levels, and the
    forecast allocates every other species its suite selects with the Registry defaults
    (mpas_atmphys_init_microphysics.F), so the suite needs nothing more from the init. The suite is still named
    in the receipt, as it is for an ArWen-route suite.
    THE BREAKAGE THIS RETIRES (hex parity run group 7, #28): the door refused the flag ("drop the flag"), so a
    caller that names the suite for both the init and the forecast, as it does on the WOOF route, stopped.
    """
    from .errors import MpasPortError
    if land_surface == "ruc":
        # THE BREAKAGE THIS PREVENTS: RUC's nine soil levels in an init whose suite's land model reads four.
        raise MpasPortError(f"--physics-suite {suite} is an MPAS-A v8.4.2 suite, whose land models (Noah, Noah-MP) "
                            "read four soil levels, and --land-surface ruc writes nine; drop --land-surface ruc")
    return {"land_surface": land_surface or "noah", "scalar_names": (),
            "physics_initialization": {
                "suite": suite, "route": "mpas-suite (MPAS-A v8.4.2)",
                "init": "case-7 fields with the land model's four soil levels, the init written with no suite",
                "species": "allocated by the forecast with the Registry defaults (mpas_atmphys_init_microphysics.F)"}}


def init_selection(suite, land_surface, *, profile=None, pin_set=None):
    from .errors import MpasPortError
    from .physics_suites import load_table, suite_row
    from .species_row import species_row_for_scheme
    table = load_table()
    if profile is not None:
        row = table.get("experimental_profiles", {}).get(profile)
        if row is None or "initialization" not in row:
            raise MpasPortError(f"physics profile {profile!r} has no initialization row")
        if suite not in (None, row["base_suite"]):
            raise MpasPortError("physics profile and initialization suite differ")
        selection = mpas_v842_selection(row["base_suite"], land_surface)
        selection.update(row["initialization"], scalar_names=tuple(row["required_init_fields"]), profile=profile)
        return selection
    if suite is None:
        return {"land_surface": land_surface or "noah", "scalar_names": ()}
    from .mpasphys.rrfs.cold_start_rrfs import producer_for_suite
    known = [row["name"] for row in table["suites"]]
    if suite not in known and producer_for_suite(suite, pin_set=pin_set) is not None:
        return producer_selection(suite, land_surface, pin_set=pin_set)
    if suite not in known and suite in set(table.get("mpasphys_v842", {}).get("suite_routes", {})):
        return mpas_v842_selection(suite, land_surface)
    if suite not in known:
        raise MpasPortError(unknown_suite_refusal(suite, known))
    selectors = suite_row(suite)["selectors"]
    land = table["options"]["config_lsm_scheme"][selectors["config_lsm_scheme"]]["carried"]["land_surface_scheme"]
    names = species_row_for_scheme(selectors["config_microp_scheme"]).names()
    required = table["options"]["config_microp_scheme"][selectors["config_microp_scheme"]].get("carried", {}).get("requires_aerosol_climatology", False)
    return {"land_surface": land_surface or land, "scalar_names": names, "requires_climatology": required}



def profile_init_matches(output, flags):
    """A cached cold init must name the requested profile's verified preparation."""
    from .global_door import forecast_arg_physics
    profile = forecast_arg_physics(flags or ()).get("profile")
    if not profile:
        return True
    path = Path(output).with_name(Path(output).name + ".provenance.json")
    if not path.is_file():
        return False
    fields = json.loads(path.read_text()).get("physics_initialization") or {}
    return (fields.get("profile") == profile
            and (fields.get("resolution") or {}).get("matches_wrf_471_reference") is True)


def _synthetic_source(path):
    receipt = Path(path).with_name(Path(path).name + ".provenance.json")
    if not receipt.is_file():
        return False
    document = json.loads(receipt.read_text())
    fields = document.get("physics_initialization") or document
    return fields.get("source") == "synthetic"


def initialize_species(engine, output, selection, arguments):
    names = selection["scalar_names"]
    argv = [str(engine), "--augment-init", str(output), "--scalar-names", ",".join(names)]
    argv.extend(("--land-surface", selection["land_surface"]))
    fields_path = None
    resolution_receipt = None
    if "nwfa" in names:
        from woof.ingest.wif_climatology import (
            resolve_wif_climatology, describe_wif_source, load_wif_climatology,
            wif_fields_for_grid,
        )
        status = aerosol_status(engine, output)
        if status is None:
            raise ValueError(_predates_suite_init(engine, "Initializing aerosol-aware microphysics (nwfa, nifa)"))
        supplied = status["analysis_pair"]
        if selection.get("requires_climatology") and (selection.get("force_climatology") or _synthetic_source(output)):
            supplied = False
        if not supplied:
            resolution = resolve_wif_climatology(getattr(arguments, "wif_climatology", None))
            if not resolution.resolved and selection.get("requires_climatology"):
                from woof.table_assets import stage_wif_dataset
                from woof.ingest.wif_dataset import resolve_wif_data_root, WIF_DATASET_FILE
                if stage_wif_dataset() != 0:
                    raise ValueError("selected microphysics requires analyzed aerosols or the verified monthly WIF dataset; acquisition failed")
                resolution = resolve_wif_climatology(resolve_wif_data_root() / WIF_DATASET_FILE)
            resolution_receipt = describe_wif_source(resolution)
            if not resolution.resolved and selection.get("requires_climatology"):
                # TEMPO's backend refuses invented aerosols at step 0
                # (woof/core/mpas_column_batch.py); a synthetic profile here
                # would only move that refusal onto a mislabelled file.
                raise ValueError("selected microphysics requires analyzed aerosols or the monthly WIF "
                                 "climatology, and neither resolved: " + json.dumps(resolution_receipt, default=str))
            if resolution.resolved:
                # Transport the target geometry to the existing Rust-seeded
                # WRF climatology pipeline as raw float32, never as text: a
                # 3 km mesh carries 2.4 million columns.
                import numpy as np
                transport = Path(output).with_name(Path(output).name + ".aerosol-transport")
                if transport.exists():
                    shutil.rmtree(transport)
                fields_path = transport
                try:
                    target_run = subprocess.run([str(engine), "--aerosol-target", str(output),
                        "--target-dir", str(transport)], capture_output=True, text=True)
                    if target_run.returncode:
                        raise ValueError("native aerosol target geometry failed: " + target_run.stderr)
                    target = json.loads(target_run.stdout)
                    cells, levels = int(target["cells"]), int(target["levels"])

                    def column(name, count):
                        values = np.fromfile(transport / target["files"][name], dtype="<f4")
                        if values.size != count:
                            raise ValueError(f"aerosol target {name} has {values.size} values, {count} expected")
                        return values

                    lat = column("latitude", cells).reshape(1, cells)
                    lon = column("longitude", cells).reshape(1, cells)
                    pressure = column("pressure", cells * levels).reshape(cells, levels).T[:, None, :]
                    phb = column("phb", cells * (levels + 1)).reshape(cells, levels + 1).T[:, None, :]
                    fields, _ = wif_fields_for_grid(load_wif_climatology(resolution.path), lat, lon,
                        arguments.start_time, np.ascontiguousarray(pressure), np.ascontiguousarray(phb),
                        backend="rust")
                    for name in ("nwfa", "nifa"):
                        values = np.ascontiguousarray(np.asarray(fields[name], dtype=np.float32)[:, 0, :].T)
                        if values.shape != (cells, levels):
                            raise ValueError(f"climatology {name} came back {values.shape}, not {(cells, levels)}")
                        values.astype("<f4").tofile(transport / f"{name}.f32le")
                except BaseException:
                    shutil.rmtree(transport, ignore_errors=True)
                    raise
                argv.extend(("--aerosol-fields", str(fields_path)))
    try:
        completed = subprocess.run(argv, capture_output=True, text=True)
    finally:
        if fields_path is not None:
            shutil.rmtree(fields_path, ignore_errors=True)
    if completed.returncode:
        raise ValueError("native physics initialization failed: " + completed.stderr)
    receipt = json.loads(completed.stdout)
    receipt["resolution"] = resolution_receipt
    receipt["argv"] = argv
    if selection.get("profile"):
        receipt["profile"] = selection["profile"]
    return receipt


def prepare_forecast_init(init, destination, physics, start_time):
    """Complete selected physics from source analysis before opening the driver.

    The point chain selects physics at forecast time. A legacy four-layer
    initializer therefore needs replay from its recorded MET, not padding.
    The caller's input stays intact; the derived file has its own receipt.
    """
    if physics is None:
        return init, None
    from netCDF4 import Dataset
    from .physics_suites import load_table
    from .species_row import species_row_for_scheme
    from .init_door import resolve_engine
    selectors = physics["selectors"]
    land = load_table()["options"]["config_lsm_scheme"][selectors["config_lsm_scheme"]].get("carried")
    if land is None:
        return init, None
    names = species_row_for_scheme(selectors["config_microp_scheme"]).names()
    # Engine-only rows (including non-aerosol Thompson) have no legacy
    # carried block. Aerosol requirements remain explicit metadata, as for TEMPO.
    required = load_table()["options"]["config_microp_scheme"][selectors["config_microp_scheme"]].get("carried", {}).get("requires_aerosol_climatology", False)
    synthetic_source = required and _synthetic_source(init)
    with Dataset(init) as ds:
        soil = len(ds.dimensions["nSoilLevels"])
        missing = set(names) - set(ds.variables)
        if start_time is None:
            if "config_start_time" in ds.ncattrs():
                start_time = str(ds.getncattr("config_start_time"))
            elif "xtime" in ds.variables:
                start_time = ds.variables["xtime"][0].tobytes().decode("ascii").strip(" \0")
    soil_change = soil != land["n_soil_levels"]
    engine = None
    aerosol_change = False
    category_change = False
    if "nwfa" in names or land["land_surface_scheme"] != "noah":
        engine = resolve_engine(None)
        state = aerosol_status(engine, init)
        if state is None:
            # THE BREAKAGE THIS PREVENTS (golden 0.4.0 round 2, install
            # gate on the woof 2.9.0 candidate b32d6fa48): every --physics
            # selection with a non-Noah land model, convection off included,
            # stopped at the door with an unnamed "Command [... rw_mpas_init
            # --aerosol-status ...] returned non-zero exit status 1"; the same
            # row ran at 64104cd, before this road existed.  Aerosols and a
            # soil-level change cannot be done without the suite-aware engine
            # and are refused by name; the shoreline soil-category check alone
            # is skipped, as the bare default run does, and said so.
            if "nwfa" in names:
                raise ValueError(_predates_suite_init(engine, f"{selectors['config_microp_scheme']} (nwfa, nifa)"))
            if soil_change:
                raise ValueError(_predates_suite_init(
                    engine, f"{land['land_surface_scheme']} with {land['n_soil_levels']} soil levels"))
            print(f"PHYSICS-INIT shoreline soil-category check not run: the installed rw_mpas_init ({engine}) "
                  f"predates {SUITE_INIT_FLAG} (woof 2.9.0's suite-aware init); the init's categories are "
                  "used as written, as a bare default run uses them", flush=True)
            return init, None
        aerosol_change = "nwfa" in names and (bool(missing) or not state["analysis_pair"] or synthetic_source)
        category_change = land["land_surface_scheme"] != "noah" and state.get("soil_category_mismatches", 0) > 0
    if not soil_change and not aerosol_change and not category_change:
        return init, None
    if not start_time:
        raise ValueError("physics initialization needs the initial state's valid time for source analysis and monthly climatology")
    engine = engine or resolve_engine(None)
    init, destination = Path(init), Path(destination)
    if destination.exists():
        raise ValueError(f"physics initialization destination already exists: {destination}")
    replay = None
    if soil_change:
        source_receipt = init.with_name(init.name + ".provenance.json")
        if not source_receipt.is_file():
            raise ValueError(f"{land['land_surface_scheme']} needs {land['n_soil_levels']} soil levels; "
                f"{init} has {soil} and no source-analysis provenance. Reinitialize from MET with the forecast suite; four-to-nine padding is not valid")
        source = json.loads(source_receipt.read_text())
        original = source["engine_argv"]
        flags = dict(zip(original[1::2], original[2::2]))
        import hashlib
        described = source["inputs"]["met"]
        met = Path(described["path"])
        if not met.is_file():
            raise ValueError(f"soil replay needs the recorded source analysis at {met}; it has been removed")
        if hashlib.sha256(met.read_bytes()).hexdigest() != described["sha256"]:
            raise ValueError(f"soil replay refuses changed source analysis: {met}")
        recorded_output = source["output"]
        if hashlib.sha256(init.read_bytes()).hexdigest() != recorded_output["sha256"]:
            raise ValueError(f"soil replay refuses changed initial state: {init}; its provenance describes different fields")
        controls = ("nfglevels", "nfgsoillevels", "extrap-airtemp", "use-spechumd",
            "theta-adv-order", "coef-3rd-order", "virtual-factor", "deep-soil-moisture",
            "landuse-table", "frac-seaice", "tsk-seaice-threshold", "oned-underflow",
            "model-name", "core-name", "model-version", "git-version")
        for key in ("--static", "--capsule", "--reference"):
            if not Path(flags[key]).is_file():
                raise ValueError(f"soil replay needs its original {key} input at {flags[key]}; it has been removed")
        replay = [str(engine), "--met", str(met), "--static", flags["--static"],
            "--capsule", flags["--capsule"], "--reference", flags["--reference"], "--out", str(destination),
            "--start-time", start_time, "--land-surface", land["land_surface_scheme"]]
        for key in controls:
            replay.extend(("--" + key, flags["--" + key]))
        completed = subprocess.run(replay, capture_output=True, text=True)
        if completed.returncode:
            raise ValueError("source-analysis soil replay failed: " + completed.stderr)
    else:
        shutil.copyfile(init, destination)
    selection = {"land_surface": land["land_surface_scheme"], "scalar_names": names, "requires_climatology": required,
                 "force_climatology": synthetic_source}
    receipt = initialize_species(engine, destination, selection,
        Namespace(start_time=start_time, wif_climatology=None))
    receipt.update(source_init=str(init), soil_levels=land["n_soil_levels"], replay_argv=replay)
    destination.with_name(destination.name + ".provenance.json").write_text(
        json.dumps(receipt, indent=2) + "\n")
    return destination, receipt
