"""Default per-forecast observations. Python schedules; Rust reads and scores."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import re
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import sys
import hashlib
import math
from concurrent.futures import ThreadPoolExecutor

SCHEMA = "gpuwm-hex.observation-scorecard/v1"
LEVELS = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10]
# The messages the native scorer reads from a reference, as .idx VAR:LEVEL selectors (NCEP naming): the
# mandatory-level height, temperature and wind, the 2 m / 10 m / surface fields and terrain; precipitation is an
# accumulation, which a file has only after its cycle's start. A row that names them is fetched as those byte
# ranges only (74 of 743 GFS 0.25 messages, 59 of 548 MB).
NCEP_IDX_SCORED_FIELDS = [f"{name}:{level} mb" for name in ("HGT", "TMP", "UGRD", "VGRD") for level in LEVELS] + [
    "TMP:2 m above ground", "DPT:2 m above ground", "RH:2 m above ground", "SPFH:2 m above ground",
    "UGRD:10 m above ground", "VGRD:10 m above ground", "PRES:surface", "HGT:surface"]
NCEP_IDX_SCORED_ACCUMULATIONS = ["APCP:surface"]
# A new reference or instrument is a metadata row, not a model-specific adapter.
REFERENCES = [
    {"label": "gfs_forecast", "model": "gfs", "source": "aws", "product": "pgrb2.0p25", "role": "forecast",
     "var_patterns": NCEP_IDX_SCORED_FIELDS, "accumulation_patterns": NCEP_IDX_SCORED_ACCUMULATIONS},
    {"label": "ifs_forecast", "model": "ifs", "source": "ecmwf", "role": "forecast",
     "precipitation_cadence_hours": 3,
     "static_terrain": True,
     "url_template": "https://storage.googleapis.com/ecmwf-open-data/{YYYYMMDD}/{HH}z/ifs/0p25/oper/{YYYYMMDD}{HH}0000-{hour}h-oper-fc.grib2",
     "mirror_templates": ["https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/{YYYYMMDD}/{HH}z/ifs/0p25/oper/{YYYYMMDD}{HH}0000-{hour}h-oper-fc.grib2"]},
    {"label": "gfs_analysis", "model": "gfs", "source": "aws", "product": "pgrb2.0p25", "role": "analysis",
     "var_patterns": NCEP_IDX_SCORED_FIELDS, "accumulation_patterns": NCEP_IDX_SCORED_ACCUMULATIONS},
    {"label": "ifs_analysis", "model": "ifs", "source": "ecmwf", "role": "analysis",
     "url_template": "https://storage.googleapis.com/ecmwf-open-data/{YYYYMMDD}/{HH}z/ifs/0p25/oper/{YYYYMMDD}{HH}0000-{hour}h-oper-fc.grib2",
     "mirror_templates": ["https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/{YYYYMMDD}/{HH}z/ifs/0p25/oper/{YYYYMMDD}{HH}0000-{hour}h-oper-fc.grib2"]},
]
QPE = "MultiSensor_QPE_01H_Pass1_00.00"
# Beside the queued scorer: the sha256 of the config file it was given (see join_automatic_scorer).
SOURCE_CONFIG = "automatic-source-config.json"
# Radiosonde archives in the order they are tried; a new archive is a row here plus its acquisition.
# NOMADS prepBUFR for the last two days; then NCEI's IGRA2 year-to-date snapshot tar that covers the run
# (every case since 2021: about 1.4 GB streamed for a 2025 globe, nothing kept but the window's rows); then
# the per-station IGRA2 files (this year's, or the period of record for the stations the run needs).
SONDE_SOURCES = ("gdas_prepbufr", "igra2_snapshot", "igra2")
REGIONAL_SONDE_SOURCES = ("gdas_prepbufr", "igra2", "igra2_snapshot")
IGRA2_ROOT = "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive"
# A period-of-record read downloads each station's whole history (an active station is 10 to 100 MB zipped).
# Breakage it prevents: the unfiltered read pulled 2,931 files (28 GB zipped, 51 GB on disk) for one 15 h run,
# filled the box's disk so its products stage failed, and held the run's card lease for 15 min (2026-10-24).
IGRA2_POR_MAX_BYTES = 8 << 30
# The verification region and the observed-precipitation source are configuration rows. Absent, the scorer
# keeps CONUS surface networks and MRMS. A run elsewhere names "region" (west/south/east/north degrees,
# optional step_degrees) and, outside MRMS coverage, "satellite_qpe" (label, variable, units, files).
CONUS = {"west": -125.0, "south": 22.0, "east": -66.0, "north": 50.0}


def rain_source(config):
    return (config.get("satellite_qpe") or {}).get("label", "mrms")


def stamp(value):
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def instant(text):
    return datetime.fromisoformat(text.replace("_", "T").replace("Z", "+00:00")).replace(tzinfo=timezone.utc)


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


class Commands:
    """Runs the scorer's subprocesses. Frames are scored on a thread pool (``score``), so the command
    index, the shared-input memo and the IEM pacing lock are guarded here."""

    def __init__(self, out, timeout, budget):
        self.out, self.timeout = out, timeout
        self.deadline = time.monotonic() + budget
        self.rows = []
        self.lock = threading.Lock()
        # rw_asos paces its own requests; two at once would double the rate the archive asked for
        # (HTTP 429 measured 2026-08-04, rw-obs asos.rs DEFAULT_REQUEST_PAUSE_MS).
        self.iem = threading.Lock()
        self.memo = {}

    def once(self, key, action):
        """One acquisition per key per scoring call, shared by every frame that needs it (the station
        list, a reference GRIB another frame already fetched, the IGRA2 archive). The bytes are the
        ones a per-frame fetch would have got; they are fetched once instead of once per frame."""
        with self.lock:
            slot = self.memo.get(key)
            owner = slot is None
            if owner:
                slot = self.memo[key] = {"done": threading.Event()}
        if owner:
            try:
                slot["value"] = action()
            except BaseException as error:
                slot["error"] = error
            finally:
                slot["done"].set()
        else:
            slot["done"].wait()
        if "error" in slot:
            raise slot["error"]
        return slot["value"]

    def run(self, argv, *, json_output=True, acquisition=True, timeout=None):
        """``timeout`` widens one acquisition past the per-command default (an archive walk of every
        radiosonde station); it is still bounded by the remaining acquisition budget."""
        remaining = self.deadline - time.monotonic()
        if acquisition and remaining <= 0:
            raise TimeoutError("scorecard acquisition budget ended; rerun woof hex score for this run")
        with self.lock:
            log = self.out / f"command-{len(self.rows):04d}.log"
            record = {"argv": list(map(str, argv)), "log": str(log)}
            self.rows.append(record)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", RAYON_NUM_THREADS="2", RUSTWX_DOWNLOAD_STREAMS="2")
        try:
            p = subprocess.run(record["argv"], capture_output=True, text=True,
                               timeout=min(timeout or self.timeout, remaining) if acquisition else max(self.timeout, 900), env=env)
            log.write_text(p.stdout + "\n" + p.stderr, encoding="utf-8")
            record["returncode"] = p.returncode
            if p.returncode:
                raise RuntimeError(f"{argv[0]} exited {p.returncode}: {p.stderr.strip()[-800:]}")
            return json.loads(p.stdout) if json_output else p.stdout
        except subprocess.TimeoutExpired as error:
            log.write_text("command timed out\n", encoding="utf-8")
            record["error"] = "timeout"
            raise TimeoutError(f"command timed out: {argv[0]}") from error
        except Exception as error:
            record["error"] = str(error)
            raise


def _once(commands, key, action):
    """``Commands.once`` when the runner shares inputs; a runner without it fetches per call."""
    return commands.once(key, action) if hasattr(commands, "once") else action()


def binary(name):
    if name == "hex_obs_scorecard":
        override = os.environ.get("WOOF_HEX_OBS_SCORECARD")
        from .helpers import path as helper_path
        candidates = [Path(override)] if override else [helper_path(name + (".exe" if os.name == "nt" else ""))]
        found = next((p for p in candidates if p.is_file()), None)
        if found is None:
            raise FileNotFoundError(
                "hex_obs_scorecard is not in this install (a build from the sdist without a woof source "
                "checkout leaves it out), so observation scoring cannot run; install the manylinux wheel, "
                "or rebuild with GPUWM_HEX_ENGINE_SOURCE=<woof checkout> and cargo on PATH, or set "
                "WOOF_HEX_OBS_SCORECARD to a built scorer (tools/build_obs_scorecard.py in the source tree)")
        return str(found.resolve())
    # rw_igra2 ships in this wheel (woof/hex/libexec, data/engine-bridges.json): the published gpuwm 2.9.0
    # wheel does not install it, so without the bundled copy every past case scored no radiosondes
    # (daily rehearsal HX7, 2026-10-15).  An explicit WOOF_RW_IGRA2 still wins, as on the engines ladder.
    if not os.environ.get("GPUWM_" + name.upper()):
        from woof.hex.engines import bundled_program
        bundled = bundled_program(name)
        if bundled is not None:
            return bundled
    from woof import bridges
    from woof.rustwx import crate_dir
    env_var, filename = "GPUWM_" + name.upper(), bridges.executable_name(name)
    found = bridges.find_artifact(env_var, filename)
    # gpuwm 2.9.3 serves rw_asos (and rw_mrms, rw_goes, ...) on demand, out of the wheel; rw_prepbufr
    # has been on demand since 2.9.1.  Without this fetch every ASOS row scored NOT_MEASURED ("rw_asos is
    # missing from the engine's Rust bridge installation", hex 0.4.1 global run on gpuwm 2.9.3, 2026-10-20),
    # while MRMS, which goes through woof.obs.frontdoor, staged its bridge.  Same step as frontdoor.require().
    fetch = getattr(bridges, "fetch_on_demand", None)
    if found is None and fetch is not None and fetch(name):
        found = bridges.find_artifact(env_var, filename)
    if found is None:
        found = next((p for p in (crate_dir() / "target/release" / name,
                                 crate_dir() / "target/debug" / name) if p.is_file()), None)
    if found is None:
        failure = getattr(bridges, "on_demand_failure", lambda _name: None)(name)
        raise FileNotFoundError(f"{name} is missing from the engine's Rust bridge installation"
                                + (f"; {failure}" if failure else ""))
    return str(found.resolve())


def discover(run, config=None):
    run = Path(run).resolve()
    config = dict(config or {})
    plan = run / "observation-scorecard" / "configuration.json"
    if not config and plan.is_file():
        config = json.loads(plan.read_text(encoding="utf-8"))
    command_path = run.parent / "command.json"
    if command_path.is_file() and command_path.stat().st_size < 10_000_000:
        argv = json.loads(command_path.read_text(encoding="utf-8")).get("command", [])
        for flag, key in [("--init", "init"), ("--grid", "grid"), ("--mesh", "mesh"), ("--start-time", "start")]:
            if flag in argv:
                value = argv[argv.index(flag) + 1]
                if key not in {"init", "grid"} or Path(value).is_absolute():
                    config.setdefault(key, value)
    receipt = run / "forecast-receipt.json"
    if receipt.is_file() and not all(config.get(key) for key in ("init", "grid", "mesh", "start")):
        r = json.loads(receipt.read_text(encoding="utf-8"))
        config.setdefault("init", r["init"]["path"])
        config.setdefault("grid", r["mesh"]["grid"])
        config.setdefault("mesh", r["mesh"]["name"])
        config.setdefault("start", r["schedule"]["start_time"])
    from .products_door import discover_histories, frame_stamp, native_request

    native = native_request(run)
    for key in ("init", "grid", "start"):
        if native.get(key):
            config.setdefault(key, native[key])
    diagnostics = {}
    templates = [str(row.get("filename") or "") for row in native.get("streams", ())
                 if row.get("name") == "diagnostics" and "output" in row.get("directions", ())]
    for template in templates or (["diag.*.nc"] if native else []):
        for path in sorted(run.glob(re.sub(r"\$[YMDhms]", "*", template))):
            if path.is_file() and frame_stamp(path):
                diagnostics[frame_stamp(path)] = path
    frames = []
    for path in discover_histories(run):
        text = frame_stamp(path)
        if text is None:
            raise ValueError(f"history frame has no valid-time stamp: {path}")
        frame = {"history": str(path), "valid_time": stamp(instant(text))}
        if text in diagnostics:
            frame["diagnostics"] = str(diagnostics[text])
        frames.append(frame)
    if config.get("verification_end_time"):
        end = instant(config["verification_end_time"])
        frames = [frame for frame in frames if instant(frame["valid_time"]) <= end]
    if not frames:
        raise ValueError("no history frames in the requested verification window")
    config.setdefault("start", frames[0]["valid_time"])
    if not config.get("init"):
        raise ValueError("run has no init provenance; give --config containing init and start")
    config.setdefault("references", REFERENCES)
    config.setdefault("thresholds_mm", [1.0, 5.0, 10.0])
    config.setdefault("widths_km", [30.0, 90.0, 180.0])
    config.setdefault("qpe_product", QPE)
    config.setdefault("prepbufr_url_template", "https://nomads.ncep.noaa.gov/pub/data/nccf/com/obsproc/prod/gdas.{YYYYMMDD}/gdas.t{HH}z.prepbufr.nr")
    # Radiosonde archives, tried in order. NOMADS keeps GDAS prepBUFR about two days, so a past case
    # falls through to IGRA2: the year-to-date snapshot that covers it, else the per-station files.
    config.setdefault("sonde_sources", list(REGIONAL_SONDE_SOURCES if config.get("region") else SONDE_SOURCES))
    config.setdefault("igra2_fetch_timeout_seconds", 180)
    config.setdefault("igra2_archive_url", IGRA2_ROOT + "/archive/")
    config.setdefault("igra2_station_list_url", IGRA2_ROOT + "/doc/igra2-station-list.txt")
    config.setdefault("igra2_por_index_url", IGRA2_ROOT + "/access/data-por/")
    config.setdefault("igra2_por_max_bytes", IGRA2_POR_MAX_BYTES)
    config.setdefault("observation_delay_hours", 7)
    config.setdefault("grib_parameters", json.loads((Path(__file__).parent / "data/observation-grib-parameters.json").read_text(encoding="utf-8")))
    config["run"] = str(run)
    return config, frames


REFERENCE_OBJECT_KEYS = ("model", "source", "product", "url_template", "mirror_templates", "var_patterns",
                         "accumulation_patterns")


def fetch_reference(row, valid, start, folder, commands):
    cycle = valid if row["role"] == "analysis" else start
    hour = int((valid - cycle).total_seconds() // 3600)
    # The same object serves several frames and roles (the IFS 0 h file is frame 0's forecast, every
    # frame's static terrain and frame 0's analysis; frame N's forecast is frame N+1's "previous").
    key = ("reference", json.dumps({k: row.get(k) for k in REFERENCE_OBJECT_KEYS}, sort_keys=True),
           stamp(cycle), hour)
    path = _once(commands, key, lambda: _fetch_reference_object(row, cycle, hour, folder, commands))
    return {**row, "path": path, "hour": hour}


def _fetch_reference_object(row, cycle, hour, folder, commands):
    folder.mkdir(parents=True, exist_ok=True)
    if row.get("url_template"):
        path = folder / "reference.grib2"
        errors = []
        for template in [row["url_template"], *row.get("mirror_templates", [])]:
            url = template.format(YYYYMMDD=cycle.strftime("%Y%m%d"), HH=f"{cycle.hour:02d}", hour=hour)
            try:
                record = commands.run([binary("hex_obs_scorecard"), "--fetch-input", url, str(path)])
                break
            except Exception as error:
                errors.append(str(error))
        else:
            raise RuntimeError("; ".join(errors))
        atomic_json(folder / "fetch.json", record)
        return str(path)
    # No --cache-dir: with one, rw_fetch keeps the object three times (its chunk cache, the assembled cache
    # copy and the named output: 1.64 GB for one 0.55 GB GFS file), and a 15 h globe's scratch held about
    # 30 GB of GFS (2026-10-24). Without one it writes the named output only. A row's var_patterns
    # are the messages the scorer reads; rw_fetch takes just those byte ranges from the .idx (auto mode
    # falls back to the whole file when the source has no .idx); its accumulation_patterns join after the
    # cycle's start (rw_fetch refuses a selector that matches nothing, and a 0 h file has no accumulation).
    argv = [binary("rw_fetch"), "fetch", "--model", row["model"], "--source", row["source"],
            "--date", cycle.strftime("%Y%m%d"), "--cycle", str(cycle.hour),
            "--hours", str(hour), "--out", str(folder)]
    if row.get("product"):
        argv += ["--product", row["product"]]
    for pattern in row.get("var_patterns", []) + (row.get("accumulation_patterns", []) if hour > 0 else []):
        argv += ["--var-pattern", pattern]
    record = commands.run(argv)
    atomic_json(folder / "fetch.json", record)
    files = record.get("files", record.get("results", []))
    paths = [x.get("path") or x.get("output") for x in files]
    path = next((p for p in paths if p and Path(p).is_file()), None)
    if path is None:
        # Different rw_fetch versions name their receipt arrays differently.
        # File discovery handles the transport, never decodes any payload.
        found = list(folder.glob("*.grib2")) + list(folder.glob("*.grb2")) + list(folder.glob("*.grib"))
        if len(found) != 1:
            raise ValueError("rw_fetch succeeded without exactly one retained GRIB file")
        path = str(found[0])
    return str(Path(path).resolve())


def acquire_sondes(valid, folder, commands, config):
    """Radiosondes valid at ``valid`` from the first archive in ``sonde_sources`` that serves them.

    Each archive that cannot is named in the reason, so a NOT_MEASURED row says which archives were tried."""
    folder.mkdir(parents=True, exist_ok=True)
    reasons = []
    for source in config.get("sonde_sources", SONDE_SOURCES):
        action = SONDE_ACQUISITIONS.get(source)
        if action is None:
            reasons.append(f"{source}: no acquisition is registered for this sonde source")
            continue
        try:
            return action(valid, folder / source, commands, config)
        except Exception as error:
            reasons.append(f"{source}: {error}")
    raise LookupError("no radiosonde archive served " + stamp(valid) + ": " + "; ".join(reasons))


def acquire_sondes_igra2(valid, folder, commands, config):
    """IGRA2 station files through the engine's rw_igra2 door, decoded by the native scorer.

    NCEI's year-to-date files hold this year's soundings and are refreshed daily (a sounding usually lands a
    day or two after launch); an earlier year is in the period-of-record files. One archive walk serves every
    frame of a scoring call: the files sit in the call's scratch and are deleted with it."""
    folder.mkdir(parents=True, exist_ok=True)
    collection = "y2d" if valid.year == datetime.now(timezone.utc).year else "por"
    shared = Path(getattr(commands, "scratch", None) or folder)
    argv = [binary("rw_igra2"), "fetch", "--collection", collection]
    region = config.get("region") or {}
    if collection == "por" or region:
        # Only the stations whose record spans this year (and that lie in the region the scorer keeps), and
        # only within the byte budget: a whole-archive read is the breakage named at IGRA2_POR_MAX_BYTES.
        box = [str(region[k]) for k in ("west", "south", "east", "north")] if region else []
        stations = _once(commands, ("igra2-stations", valid.year, tuple(box)), lambda: commands.run(
            [binary("hex_obs_scorecard"), "--igra2-station-ids", config.get("igra2_station_list_url", IGRA2_ROOT + "/doc/igra2-station-list.txt"),
             config.get("igra2_por_index_url", IGRA2_ROOT + "/access/data-por/"), str(valid.year), *box]))
        budget = int(config.get("igra2_por_max_bytes", IGRA2_POR_MAX_BYTES))
        if not stations.get("stations"):
            raise LookupError(f"the IGRA2 station list has no station reporting in {valid.year}"
                              + (" inside the verification region" if box else ""))
        if collection == "por" and stations["bytes"] > budget:
            raise LookupError(f"IGRA2 keeps {valid.year} soundings only in per-station period-of-record files: the "
                              f"{len(stations['stations'])} stations this run needs are {stations['bytes'] / 2**30:.1f} GiB "
                              f"to download, over igra2_por_max_bytes ({budget / 2**30:.1f} GiB); raise it in the "
                              "observation config to read them")
        argv += ["--stations", ",".join(stations["stations"])]
        archive = shared / f"igra2-{collection}-{valid.year}-{hashlib.sha256(','.join(stations['stations']).encode()).hexdigest()[:12]}"
    else:
        archive = shared / f"igra2-{collection}"
    record_path = archive / "fetch.json"

    def walk():
        if record_path.is_file():
            return json.loads(record_path.read_text(encoding="utf-8"))
        return commands.run([*argv, "--out", str(archive)],
                            timeout=float(config.get("igra2_fetch_timeout_seconds", 180)))
    # Frames run in parallel: one archive walk, never one per 00Z/12Z frame at once.
    record = _once(commands, ("igra2", str(archive)), walk)
    files = [row["path"] for row in record.get("files", []) if row.get("path")]
    if not files:
        raise LookupError(f"rw_igra2 fetched no {collection} station files")
    table = folder / "igra2-sondes.csv"
    window = (stamp(valid - timedelta(hours=1)), stamp(valid + timedelta(hours=1)))
    decoded = commands.run([binary("hex_obs_scorecard"), "--decode-igra", str(table), *window, *files])
    atomic_json(folder / "decode.json", {**decoded, "collection": collection, "station_files": len(files),
                                         "archive": record.get("archive")})
    if not decoded.get("soundings"):
        raise LookupError(f"IGRA2 {collection} holds no sounding within an hour of {stamp(valid)} yet "
                          "(NCEI refreshes the year-to-date files daily)")
    return str(table)


def acquire_sondes_igra2_snapshot(valid, folder, commands, config):
    """IGRA2 soundings from the NCEI year-to-date snapshot tar that covers the scoring call's sonde frames.

    NCEI keeps each year's year-to-date files as one tar (``archive/IGRA_v2.2_data-y2d_s..._e..._c....tar``,
    every station, about 1.4 GB for 2025-01 to 2026-07). The native scorer streams it once per scoring call in
    parallel byte ranges, decodes each station in memory on every core and writes only the soundings inside
    the call's sonde window, so no archive lands on the disk. One stream serves every frame of the call."""
    folder.mkdir(parents=True, exist_ok=True)
    first, last = getattr(commands, "sonde_window", None) or (valid, valid)
    if not first <= valid <= last:
        first = last = valid
    start, end = stamp(first - timedelta(hours=1)), stamp(last + timedelta(hours=1))
    shared = Path(getattr(commands, "scratch", None) or folder)
    shared.mkdir(parents=True, exist_ok=True)
    table = shared / f"igra2-snapshot-{first:%Y%m%d%H}-{last:%Y%m%d%H}.csv"
    url = config.get("igra2_archive_url", IGRA2_ROOT + "/archive/")
    record_path = table.with_suffix(".json")

    def stream():
        if record_path.is_file():
            return json.loads(record_path.read_text(encoding="utf-8"))
        record = commands.run([binary("hex_obs_scorecard"), "--fetch-igra2-snapshot", url, start, end, str(table)],
                              timeout=float(config.get("igra2_fetch_timeout_seconds", 180)))
        atomic_json(record_path, record)
        return record
    record = _once(commands, ("igra2-snapshot", url, start, end), stream)
    atomic_json(folder / "decode.json", record)
    near = sum(count for hour, count in record.get("soundings_by_nominal_hour", {}).items()
               if abs((instant(hour) - valid).total_seconds()) <= 3600)
    if not near:
        raise LookupError(f"IGRA2 snapshot {record.get('snapshot', {}).get('name')} holds no sounding within an hour "
                          f"of {stamp(valid)}")
    return str(table)


def acquire_sondes_prepbufr(valid, folder, commands, config):
    """GDAS prepBUFR (NOMADS keeps about two days), decoded by the native scorer."""
    folder.mkdir(parents=True, exist_ok=True)
    if config.get("prepbufr_url_template"):
        # Global GDAS exceeds the prepBUFR door's 64 MiB response bound.
        # Reuse the engine's native HTTP agent with a streaming bounded copy.
        # rw_fetch object is GRIB-only; decode remains rw_prepbufr.
        path = folder / f"gdas.t{valid.hour:02d}z.prepbufr.nr"
        url = config["prepbufr_url_template"].format(YYYYMMDD=valid.strftime("%Y%m%d"), HH=f"{valid.hour:02d}")
        record = commands.run([binary("hex_obs_scorecard"), "--fetch-input", url, str(path)])
    else:
        record = commands.run([binary("rw_prepbufr"), "fetch", "--cycle", stamp(valid), "--out", str(folder)])
        path = record.get("path") or next(iter(folder.glob("*.nr")), None)
    atomic_json(folder / "fetch.json", record)
    if path is None:
        raise ValueError("prepBUFR fetch returned no raw object")
    listing = folder / "listing.txt"
    decoded = commands.run([binary("hex_obs_scorecard"), "--decode-sondes", str(path), str(listing)])
    atomic_json(folder / "decode.json", decoded)
    return str(listing)


SONDE_ACQUISITIONS = {"gdas_prepbufr": acquire_sondes_prepbufr, "igra2_snapshot": acquire_sondes_igra2_snapshot,
                      "igra2": acquire_sondes_igra2}


SURFACE_HALF_WINDOW = timedelta(minutes=10)


def acquire_surface(valid, folder, commands, region=None):
    """ASOS/METAR reports within ten minutes of ``valid`` as a neutral table.

    The station list (one IEM metadata request per network: 266 for the globe, about 40 s) is the same
    for every frame and is fetched once per scoring call. The report window stays one request per frame,
    exactly the request the serial scorer made: IEM answers that query byte for byte the same each time
    (measured 08:05Z and 09:40Z 2026-10-16, sha256 098341c7...), while a wider window that is cut per
    frame returns two reports for the same station and minute in another order, and ``rw_asos table``
    keeps the first, so 2 of 4,855 frame-0 reports changed (measured, PASH and PRX 23:55Z). IEM requests
    run one at a time at the archive's own pace (``Commands.iem``); every other acquisition runs beside them."""
    from woof.obs.surface_networks import networks_for_bbox
    folder.mkdir(parents=True, exist_ok=True)
    raw, decoded = folder / "raw.csv", folder / "table.csv"
    shared = Path(getattr(commands, "scratch", None) or folder)
    asos = binary("rw_asos")
    box = region or CONUS
    west, south, east, north = (box[k] for k in ("west", "south", "east", "north"))
    networks = ",".join(networks_for_bbox(west, south, east, north))
    bbox = f"{west:g},{south:g},{east:g},{north:g}"
    pace = getattr(commands, "iem", None) or threading.Lock()

    def stations():
        path = shared / ("asos-stations-" + hashlib.sha256(f"{networks}|{bbox}".encode()).hexdigest()[:12] + ".json")
        with pace:
            commands.run([asos, "stations", "--networks", networks, "--bbox", bbox, "--out", str(path)])
        return path
    table = _once(commands, ("asos-stations", networks, bbox), stations)
    with pace:
        commands.run([asos, "fetch", "--stations", str(table), "--start", stamp(valid - SURFACE_HALF_WINDOW),
                      "--end", stamp(valid + SURFACE_HALF_WINDOW), "--out", str(raw)])
    commands.run([asos, "table", "--obs", str(raw), "--stations", str(table), "--out", str(decoded)])
    return str(decoded)


_SHARED_LOCK = threading.Lock()


def _drop_tree(path):
    """Delete an acquisition's raw objects once they are decoded; the scorer reads only the decoded files."""
    if path.is_dir():
        for item in sorted(path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            item.unlink() if item.is_file() or item.is_symlink() else item.rmdir()
        path.rmdir()


def _share_identical(commands, path):
    """Keep one inode for byte-identical decoded files (an MRMS grid is 264 MB and the same every hour):
    the first copy is kept and each identical later one becomes a hard link to it."""
    import filecmp
    with _SHARED_LOCK:
        kept = commands.__dict__.setdefault("shared_files", {}).setdefault(path.name, [])
        for first in kept:
            if first.stat().st_size == path.stat().st_size and filecmp.cmp(first, path, shallow=False):
                temp = path.with_name(path.name + ".link")
                os.link(first, temp)
                os.replace(temp, path)
                return
        kept.append(path)


def acquire_rain(valid, previous, folder, commands, product):
    from woof.verification_visuals import fetch_radar
    duration = int((valid - previous).total_seconds())
    if duration <= 0 or duration % 3600:
        raise ValueError("MRMS comparison needs a positive whole-hour accumulation interval")
    # Reuse the engine acquisition door, including pack QC and geometry. The hours are independent
    # objects, fetched at once and kept in hour order. Each hour is fetched once per scoring call into the
    # call's scratch (frames 3 h apart share two of their three hours), its raw object is dropped once
    # decoded, and the grid, the same for every hour of a product, is kept once.
    shared = Path(getattr(commands, "scratch", None) or folder) / "mrms"

    def fetch(when):
        target = shared / f"{product}-{when:%Y%m%dT%H%M}"
        row = fetch_radar({"valid_time": stamp(when)}, {"product": product, "key": "precipitation_1h"},
                          (-125, 22, -66, 50), target, timeout=min(commands.timeout, max(1, commands.deadline - time.monotonic())))
        _drop_tree(target / "objects")
        _share_identical(commands, Path(row["grid_path"]))
        return row

    def one(hour):
        when = previous + timedelta(hours=hour)
        row = _once(commands, ("mrms", product, stamp(when)), lambda: fetch(when))
        return {**row, "valid_time": stamp(when)}
    hours = range(1, duration // 3600 + 1)
    with ThreadPoolExecutor(max_workers=len(hours)) as pool:
        return list(pool.map(one, hours))


SONDE_NOT_APPLICABLE = "radiosondes launch at 00Z and 12Z only; no sounding is valid at this frame"


def sonde_applies(valid):
    """Radiosondes are launched at 00Z and 12Z. At any other valid time a sonde cell cannot apply, so
    its rows are NOT_APPLICABLE and count neither for nor against coverage."""
    return valid.hour in (0, 12)


def missing_rows(frame, source, reason, config, status="NOT_MEASURED"):
    labels = ["hex"] + [r["label"] for r in config["references"]]
    quantities = ["height", "temperature", "wind"] if source == "radiosonde" else ["temperature", "dewpoint", "wind", "pressure"]
    if source == rain_source(config):
        return [{"source": source, "quantity": "precipitation", "label": label, "valid_time": frame["valid_time"],
                 "threshold_mm": threshold, "width_km": width, "status": status, "value": None, "n": 0, "reason": reason}
                for label in ["hex"] + [r["label"] for r in config["references"] if r["role"] == "forecast"]
                for threshold in config["thresholds_mm"] for width in config["widths_km"]]
    return [{"source": source, "quantity": quantity, "label": label, "valid_time": frame["valid_time"], "level_hpa": level,
             "status": status, "rmse": None, "bias": None, "n": 0, "reason": reason}
            for label in labels for level in (LEVELS if source == "radiosonde" else [0]) for quantity in quantities]


def markdown(card):
    rows = card["metrics"]
    labels = ["hex"] + [r["label"] for r in card["configuration"]["references"]]
    def cell(source, quantity, label, level=None):
        targets = [r["valid_time"] for r in rows if r["source"] == source and r["quantity"] == quantity
                   and r["label"] == "hex" and r["status"] == "MEASURED"
                   and (level is None or r.get("level_hpa") == level)]
        if not targets:
            return "N/M"
        target = max(targets)
        found = [r for r in rows if r["source"] == source and r["quantity"] == quantity and r["label"] == label
                 and (level is None or r.get("level_hpa") == level) and r["status"] == "MEASURED" and r["valid_time"] == target]
        if not found:
            return "N/M"
        # One page reports the latest measured lead, not an invalid average of RMSEs.
        r = max(found, key=lambda r: r["valid_time"])
        return f"{r['rmse']:.2f} ({r['n']})"
    text = [f"# {Path(card['run']).parent.name}: observation scorecard", "",
            f"Coverage: **{card['status']}**. Initialization: {card['configuration']['start']}. Latest measured hex lead per row; all columns use that same valid time. JSON carries every lead and missing reason.", "",
            "Radiosonde RMSE: height m / temperature K / vector wind m s-1. Counts follow each value. Analyses are secondary references and may assimilate these reports.", "",
            "| hPa | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for level in LEVELS:
        text.append(f"| {level} | " + " | ".join(" / ".join(cell("radiosonde", q, label, level) for q in ("height", "temperature", "wind")) for label in labels) + " |")
    text += ["", "| ASOS RMSE | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for q, unit in [("temperature", "T2 K"), ("dewpoint", "Td2 K"), ("wind", "vector wind m s-1"), ("pressure", "station pressure Pa")]:
        text.append("| " + unit + " | " + " | ".join(cell("asos", q, label) for label in labels) + " |")
    rain_labels = ["hex"] + [r["label"] for r in card["configuration"]["references"] if r["role"] == "forecast"]
    rs = rain_source(card["configuration"])
    text += ["", ("MRMS" if rs == "mrms" else rs) + " accumulation FSS. Thresholds are interval totals in mm; neighborhoods are full widths in km.", "",
             "| mm / km | " + " | ".join(rain_labels) + " |", "|---|" + "---|" * len(rain_labels)]
    for threshold in card["configuration"]["thresholds_mm"]:
        for width in card["configuration"]["widths_km"]:
            values = []
            targets = [r["valid_time"] for r in rows if r["source"] == rs and r["label"] == "hex"
                       and r.get("threshold_mm") == threshold and r.get("width_km") == width and r["status"] == "MEASURED"]
            target = max(targets) if targets else None
            for label in rain_labels:
                found = [r for r in rows if r["source"] == rs and r["label"] == label and r.get("threshold_mm") == threshold and r.get("width_km") == width and r["status"] == "MEASURED" and r["valid_time"] == target]
                values.append(f"{found[-1]['value']:.3f} ({found[-1]['interval_hours']} h)" if found else "N/M")
            text.append(f"| {threshold:g} / {width:g} | " + " | ".join(values) + " |")
    reasons = list(dict.fromkeys(r["reason"] for r in rows if r.get("reason") and r["status"] != "NOT_APPLICABLE"))
    text += ["", "N/M reasons: " + "; ".join(reasons[:6]) + ("; see JSON for additional reasons" if len(reasons) > 6 else ""),
             "", "Paired diagnostics cluster squared-error differences by station. Single-cycle results are exploratory; no independent-event significance or model promotion is claimed."]
    return "\n".join(text) + "\n"


SCORER_BYTES = 8 << 30  # one native scorer at its peak: 5.3 GiB measured (60 km globe, 4 global references)


def score_workers():
    """Frames scored at once. WOOF_HEX_SCORE_WORKERS overrides; the default is one per two cores, at
    most 16, and no more than the memory the host has available holds native scorers (SCORER_BYTES each),
    so a workstation scores fewer frames at once instead of running out of memory."""
    raw = os.environ.get("WOOF_HEX_SCORE_WORKERS")
    if raw:
        return max(1, int(raw))
    workers = max(1, min(16, (os.cpu_count() or 2) // 2))
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            available = next(int(line.split()[1]) * 1024 for line in handle if line.startswith("MemAvailable:"))
        workers = min(workers, max(1, available // SCORER_BYTES))
    except (OSError, StopIteration, ValueError):
        pass
    return workers


def _score_frame(index, frame, context):
    """One frame's acquisitions and native score. Returns its rows; nothing here touches the card."""
    try:
        return _score_frame_rows(index, frame, context)
    except Exception as error:
        return {"error": error}


def _score_frame_rows(index, frame, context):
    config, frames, start, commands = context["config"], context["frames"], context["start"], context["commands"]
    rs = context["rs"]
    sources = ("radiosonde", "asos", rs)
    rows = {"metrics": [], "paired": [], "sources": []}
    valid = instant(frame["valid_time"])
    sonde_ok = sonde_applies(valid)
    if valid > datetime.now(timezone.utc):
        for source in sources:
            if source == "radiosonde" and not sonde_ok:
                rows["metrics"].extend(missing_rows(frame, source, SONDE_NOT_APPLICABLE, config, "NOT_APPLICABLE"))
            else:
                rows["metrics"].extend(missing_rows(frame, source, "observations pending for a future valid time", config))
        return rows
    folder = context["scratch"] / str(index)
    folder.mkdir()
    request = {**frame, "start": stamp(start), "init": config["init"], "references": [], "thresholds_mm": config["thresholds_mm"], "widths_km": config["widths_km"], "grib_parameters": config["grib_parameters"]}
    if config.get("region"):
        request["region"] = config["region"]
    local_files = config.get("observation_files", {}).get(frame["valid_time"], {})

    def sondes():
        return local_files.get("sondes") or acquire_sondes(valid, folder / "radiosonde", commands, config)

    def surface():
        return local_files.get("surface") or acquire_surface(valid, folder / "asos", commands, config.get("region"))

    previous, cadence_missing = None, None
    if index > 0:
        duration = int((valid - instant(frames[index - 1]["valid_time"])).total_seconds() // 3600)
        cadence = math.lcm(duration, *[int(r.get("precipitation_cadence_hours", 1)) for r in config["references"] if r["role"] == "forecast"])
        cadence = int(config.get("precipitation_interval_hours", cadence))
        previous = next((f for f in frames[:index] if instant(f["valid_time"]) == valid - timedelta(hours=cadence)), None)
        if previous is None:
            cadence_missing = {"source": rs, "reason": f"no history boundary {cadence} h earlier for a common reference precipitation interval"}
            previous = frames[index - 1]
        request["previous_history"] = previous["history"]
        if previous.get("diagnostics"):
            request["previous_diagnostics"] = previous["diagnostics"]
        request["previous_time"] = previous["valid_time"]

    def rain():
        if cadence_missing is not None:
            raise LookupError(cadence_missing["reason"])
        if config.get("satellite_qpe"):
            # Rust selects the slices that tile (previous, valid] and refuses a partial tiling.
            return ("qpe", dict(config["satellite_qpe"]))
        return ("rain", local_files.get("rain") or acquire_rain(valid, instant(previous["valid_time"]), folder / "mrms", commands, config["qpe_product"]))

    def reference(row):
        missed = []
        try:
            local = config.get("reference_files", {}).get(frame["valid_time"], {}).get(row["label"])
            value = {**row, **local} if local else fetch_reference(row, valid, start, folder / row["label"], commands)
            if row.get("static_terrain") and row["role"] == "forecast" and not value.get("static_path"):
                try:
                    static = fetch_reference({**row, "role": "analysis"}, start, start, folder / (row["label"] + "-static"), commands)
                    value["static_path"] = static["path"]
                except Exception as error:
                    missed.append({"source": "asos", "label": row["label"], "reason": "static terrain: " + str(error)})
            # Failure of an optional preceding accumulation must not discard
            # the current reference's measured temperature/profile fields.
            if index > 0 and row["role"] == "forecast" and instant(request["previous_time"]) > start and not local:
                try:
                    prior = fetch_reference(row, instant(request["previous_time"]), start, folder / (row["label"] + "-previous"), commands)
                    value["previous_path"] = prior["path"]
                except Exception as error:
                    missed.append({"source": rs, "label": row["label"], "reason": str(error)})
            return value, missed
        except Exception as error:
            return None, missed + [{"source": row["label"], "reason": str(error)}]

    def caught(action):
        try:
            return action(), None
        except Exception as error:
            return None, error

    # Every acquisition of this frame at once; the request and the missing list are then assembled in
    # the order the serial scorer used, so reasons and rows come out the same.
    references = config["references"]
    with ThreadPoolExecutor(max_workers=3 + len(references)) as pool:
        # No sounding exists off 00Z/12Z, so nothing is fetched for it (IGRA2/prepbufr are not tried).
        got_sondes = pool.submit(caught, sondes) if sonde_ok else None
        got_surface = pool.submit(caught, surface)
        got_rain = pool.submit(caught, rain) if index > 0 else None
        got_references = [pool.submit(reference, row) for row in references]
        missing = []
        for source, key, future in [("radiosonde", "sondes", got_sondes), ("asos", "surface", got_surface)]:
            if future is None:
                continue
            value, error = future.result()
            if error is not None:
                missing.append({"source": source, "reason": str(error)})
            else:
                request[key] = value
        if index > 0:
            if cadence_missing is not None:
                missing.append(cadence_missing)
            value, error = got_rain.result()
            if error is not None:
                missing.append({"source": rs, "reason": str(error)})
            else:
                request[value[0]] = value[1]
        else:
            missing.append({"source": rs, "reason": "initial frame has no forecast accumulation interval"})
        for future in got_references:
            value, missed = future.result()
            missing.extend(missed)
            if value is not None:
                request["references"].append(value)
    out, native = context["out"], context["native"]
    req, result = out / f"request-{index:03d}.json", out / f"native-{index:03d}.json"
    atomic_json(req, request)
    try:
        commands.run([native, str(req), str(result)], json_output=False, acquisition=False)
        receipt = json.loads(result.read_text(encoding="utf-8"))
        rows["sources"].append(receipt["provenance"])
        rows["paired"].extend({**row, "valid_time": frame["valid_time"]} for row in receipt["paired"])
        rows["metrics"].extend({**row, "valid_time": frame["valid_time"]} for row in receipt["metrics"])
        missing.extend(receipt["failures"])
    except Exception as error:
        missing += [{"source": s, "reason": str(error)} for s in sources]
    for source in sources:
        for miss in missing:
            if miss.get("source") == source and not miss.get("label"):
                rows["metrics"] = [r for r in rows["metrics"] if r["source"] != source]
                rows["metrics"].extend(missing_rows(frame, source, miss["reason"], config))
    if not sonde_ok:
        # The native scorer, given no soundings, emits empty radiosonde rows; at this frame they cannot apply.
        rows["metrics"] = [r for r in rows["metrics"] if r["source"] != "radiosonde"]
        rows["paired"] = [r for r in rows["paired"] if r.get("source") != "radiosonde"]
        rows["metrics"].extend(missing_rows(frame, "radiosonde", SONDE_NOT_APPLICABLE, config, "NOT_APPLICABLE"))
    for row in references:
        for source in sources:
            # Absent references remain explicit even when hex was measured.
            if not any(r["source"] == source and r["label"] == row["label"] for r in rows["metrics"]):
                reason = next((m["reason"] for m in missing if m.get("source", m.get("label")) == row["label"]), "reference quantity not supplied")
                rows["metrics"].extend(r for r in missing_rows(frame, source, reason, config) if r["label"] == row["label"])
    atomic_json(out / f"missing-{index:03d}.json", missing)
    return rows


def score(run, *, config=None, out=None, timeout=180, budget=1800, leads=None):
    config, frames = discover(run, config)
    out = Path(out or Path(run) / "observation-scorecard").resolve()
    out.mkdir(parents=True, exist_ok=True)
    from woof.verification_visuals import _run_lock
    lock = _run_lock(out, name=".hex-scorecard.lock")
    lock.__enter__()
    scratch = None
    commands = Commands(out, timeout, budget)
    card = {"schema": SCHEMA, "run": config["run"], "configuration": config, "metrics": [], "paired": [], "sources": [], "commands": commands.rows, "deleted_files": []}
    try:
        atomic_json(out / "configuration.json", config)
        scratch = Path(tempfile.mkdtemp(prefix=".inputs-", dir=out))
        commands.scratch = scratch  # archives shared by every frame of this call (IGRA2 station files)
        start = instant(config["start"])
        native = binary("hex_obs_scorecard")
        with open(native, "rb") as handle:
            card["native_binary_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
        previous_path = out / "scorecard.json"
        if leads is not None and previous_path.is_file():
            prior = json.loads(previous_path.read_text(encoding="utf-8"))
            compatible = prior.get("native_binary_sha256") == card["native_binary_sha256"]
            compatible |= prior.get("status") == "QUEUED" and not prior.get("measured_metrics")
            if prior.get("configuration") == config and compatible:
                selected = {f["valid_time"] for f in frames if int((instant(f["valid_time"]) - start).total_seconds() // 3600) in leads}
                card["metrics"] = [r for r in prior["metrics"] if r["valid_time"] not in selected]
                card["paired"] = [r for r in prior["paired"] if r["valid_time"] not in selected]
                card["sources"] = prior["sources"]
                card["deleted_files"] = prior["deleted_files"]
                commands.rows.extend(prior["commands"])
        abi = commands.run([native, "--abi"], json_output=False, acquisition=False).strip()
        if abi != "gpuwm-hex.native-observation-score/v1":
            raise ValueError("hex observation binary ABI differs from its scoring contract")
        context = {"config": config, "frames": frames, "start": start, "scratch": scratch, "out": out,
                   "commands": commands, "native": native, "rs": rain_source(config)}
        todo = []
        for index, frame in enumerate(frames):
            lead = int((instant(frame["valid_time"]) - start).total_seconds() // 3600)
            if leads is None or lead in leads:
                todo.append((index, frame))
        # The sonde frames this call scores: one IGRA2 snapshot stream decodes all of them at once.
        sonde_times = [instant(f["valid_time"]) for _, f in todo if sonde_applies(instant(f["valid_time"]))
                       and instant(f["valid_time"]) <= datetime.now(timezone.utc)]
        if sonde_times:
            commands.sonde_window = (min(sonde_times), max(sonde_times))
        # Frames are independent until their rows join the card, so they are scored at once (the
        # acquisitions are network waits, the native scorer two threads each) and joined in frame order:
        # the card is the one a frame-by-frame loop writes.
        workers = max(1, min(len(todo), score_workers()))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hex-score") as pool:
            results = list(pool.map(lambda item: _score_frame(item[0], item[1], context), todo))
        for result in results:
            if result.get("error") is not None:
                raise result["error"]
            card["sources"].extend(result["sources"])
            card["paired"].extend(result["paired"])
            card["metrics"].extend(result["metrics"])
    except Exception as error:
        card["error"] = f"{type(error).__name__}: {error}"
    finally:
        if scratch is not None:
            for path in sorted(scratch.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if path.is_file() or path.is_symlink():
                    card["deleted_files"].append({"path": str(path), "bytes": path.lstat().st_size})
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            scratch.rmdir()
        # Coverage counts only cells that can apply: NOT_APPLICABLE rows (sondes off 00Z/12Z) are neither
        # measured nor missing.
        applicable = [r for r in card["metrics"] if r["status"] != "NOT_APPLICABLE"]
        measured = sum(r["status"] == "MEASURED" for r in applicable)
        card["status"] = "MEASURED" if measured and measured == len(applicable) and not card.get("error") else "PARTIALLY_MEASURED" if measured else "NOT_MEASURED"
        card["measured_metrics"] = measured
        card["applicable_metrics"] = len(applicable)
        card["not_applicable_metrics"] = len(card["metrics"]) - len(applicable)
        atomic_json(out / "scorecard.json", card)
        (out / "scorecard.md").write_text(markdown(card), encoding="utf-8")
        lock.__exit__(None, None, None)
    return card


def future_frames(frames, config, now=None):
    """Frames whose valid time has not happened yet, so no observation of them can exist.

    The automatic scorer never waits for these. Breakage it prevents: a 3 km region forecast from GFS f348
    (valid 13 days ahead) left its watcher sleeping in the leased command's tree for 75 min, until it was
    killed by PID, and the run's products step waited behind it (box L, 2026-10-24, INVENTORY F-SCORE-LEASE).
    Each frame gets the time its observations can first be scored and the command that scores it then."""
    now = now or datetime.now(timezone.utc)
    delay = timedelta(hours=float(config.get("observation_delay_hours", 7)))
    return {frame["valid_time"]: stamp(instant(frame["valid_time"]) + delay)
            for frame in frames if instant(frame["valid_time"]) > now}


def future_reason(valid_time, ready_at, run):
    return (f"valid {valid_time} is in the future, so no observations exist yet; they can be scored from "
            f"{ready_at} with: woof hex score {run}")


#: Set to "off" by a door run with ``--skip score``: the forecast queues no scorer (no watcher, no downloads).
SCORING_ENV = "WOOF_HEX_OBSERVATION_SCORING"


def forecast_hook(output, *, init, grid, start, config_path=None):
    """Schedule CPU scoring without retaining a forecast's GPU lease."""
    if str(os.environ.get(SCORING_ENV, "")).strip().lower() == "off":
        # `--skip score` on a door: the run asked for no scoring, so no watcher is started and nothing is
        # downloaded (it had still started the watcher and its IGRA2 period-of-record download, ~50 GB over
        # four 1 h runs on one box). The record says so instead of leaving the folder silent.
        target = Path(output) / "observation-scorecard"
        target.mkdir(parents=True, exist_ok=True)
        record = {"schema": SCHEMA, "run": str(output), "status": "SKIPPED",
                  "reason": f"{SCORING_ENV}=off (the door was run with --skip score)", "measured_metrics": 0}
        atomic_json(target / "scorecard.json", record)
        (target / "scorecard.md").write_text(f"# Observation scorecard\n\nSKIPPED: {record['reason']}\n",
                                             encoding="utf-8")
        return {"status": "SKIPPED", "path": str(target / "scorecard.json")}
    try:
        output = Path(output)
        provided = Path(config_path or os.environ.get("WOOF_HEX_OBSERVATION_CONFIG") or output / "observation-scorecard.config.json")
        source = provided.read_bytes() if provided.is_file() else None
        config = json.loads(source.decode("utf-8")) if source is not None else {}
        config.update(init=str(init), grid=str(grid), start=str(start))
        config, frames = discover(output, config)
        target = output / "observation-scorecard"
        target.mkdir(parents=True, exist_ok=True)
        owner = config.get("verification_owner")
        if owner:
            # The S2S coordinator merges restart segments and scores its bounded
            # ten-day view. A second future waiter would pin disposable history
            # for weeks and score incomplete segments independently.
            rows = [row for frame in frames for source in ("radiosonde", "asos", rain_source(config))
                    for row in missing_rows(frame, source, f"verification owned by {owner}", config)]
            card = {"schema": SCHEMA, "run": str(output), "configuration": config,
                    "status": "DEFERRED", "metrics": rows, "measured_metrics": 0,
                    "paired": [], "sources": [], "commands": [], "deleted_files": []}
            atomic_json(target / "scorecard.json", card)
            (target / "scorecard.md").write_text(markdown(card), encoding="utf-8")
            return {"status": "DEFERRED", "owner": owner, "path": str(target / "scorecard.json")}
        future = future_frames(frames, config)
        reasons = {valid: future_reason(valid, ready_at, output) for valid, ready_at in future.items()}
        if len(future) == len(frames):
            # Every frame is valid in the future: start no watcher and retain nothing. The record says why and
            # when to score; the receipt lets a door's score stage report it at once (join_automatic_scorer).
            rows = [row for frame in frames for source in ("radiosonde", "asos", rain_source(config))
                    for row in missing_rows(frame, source, reasons[frame["valid_time"]], config)]
            first = min(future.values())
            reason = (f"every frame is valid in the future (last {frames[-1]['valid_time']}); no observations "
                      f"exist yet, so no automatic scorer was started. Score from {first} (all frames from "
                      f"{max(future.values())}) with: woof hex score {output}")
            card = {"schema": SCHEMA, "run": str(output), "configuration": config, "status": "NOT_YET_OBSERVABLE",
                    "reason": reason, "observable_from": future, "metrics": rows, "measured_metrics": 0,
                    "paired": [], "sources": [], "commands": [], "deleted_files": []}
            atomic_json(target / "configuration.json", config)
            atomic_json(target / "scorecard.json", card)
            (target / "scorecard.md").write_text(markdown(card) + f"\nNOT_YET_OBSERVABLE: {reason}\n", encoding="utf-8")
            atomic_json(target / SOURCE_CONFIG, {"path": str(provided) if source is not None else None,
                                                 "sha256": hashlib.sha256(source).hexdigest() if source is not None else None})
            atomic_json(target / "automatic-retry-receipt.json", {"status": "NOT_YET_OBSERVABLE", "pid": None,
                                                                  "reason": reason, "observable_from": future})
            print(f"OBSERVATIONS NOT YET OBSERVABLE {target / 'scorecard.json'}: {reason}", flush=True)
            return {"status": "NOT_YET_OBSERVABLE", "path": str(target / "scorecard.json"), "reason": reason}
        # Hold one shared inode, not a second prepared copy, while future
        # reports arrive. The worker releases its own link and marker.
        pin = target / "pending-init.nc"
        try:
            if not pin.exists():
                os.link(Path(init).resolve(), pin)
            config["init"] = str(pin.resolve())
            config["owned_init_pin"] = str(pin)
        except OSError as error:
            config["input_retention_warning"] = f"init hardlink unavailable: {error}; original init must remain until scoring finishes"
        marker = output / ".keep"
        if not marker.exists():
            marker.write_text("hex observation scoring pending: retain these specific inputs until the automatic retry receipt is written\n" +
                              "\n".join([f["history"] for f in frames] + [config["init"]]) + "\n", encoding="utf-8")
            config["owned_keep_marker"] = str(marker)
        rows = [row for frame in frames for source in ("radiosonde", "asos", rain_source(config))
                for row in missing_rows(frame, source, reasons.get(frame["valid_time"], "automatic observation scoring queued"), config)]
        card = {"schema": SCHEMA, "run": str(output), "configuration": config, "status": "QUEUED", "metrics": rows,
                "measured_metrics": 0, "paired": [], "sources": [], "commands": [], "deleted_files": []}
        atomic_json(target / "scorecard.json", card)
        (target / "scorecard.md").write_text(markdown(card), encoding="utf-8")
        # Which config file the queued scorer was given, so a door's own score stage can tell it is
        # the same request and join it instead of scoring the run twice (join_automatic_scorer).
        atomic_json(target / SOURCE_CONFIG, {"path": str(provided) if source is not None else None,
                                             "sha256": hashlib.sha256(source).hexdigest() if source is not None else None})
        queue_retry(output, config)
        print(f"OBSERVATIONS QUEUED {target / 'scorecard.json'}", flush=True)
        return {"status": "QUEUED", "path": str(target / "scorecard.json")}
    except BlockingIOError:
        return {"status": "IN_PROGRESS", "path": str(Path(output) / "observation-scorecard/scorecard.json")}
    except Exception as error:
        # A missing history/output is not permission to silently omit the hook.
        target = Path(output) / "observation-scorecard"
        target.mkdir(parents=True, exist_ok=True)
        record = {"schema": SCHEMA, "status": "NOT_MEASURED", "error": f"{type(error).__name__}: {error}"}
        atomic_json(target / "scorecard.json", record)
        (target / "scorecard.md").write_text(f"# Observation scorecard\n\nNOT_MEASURED: {record['error']}\n", encoding="utf-8")
        return {"status": "NOT_MEASURED", "path": str(target / "scorecard.json"), "reason": record["error"]}


def add_score_parser(commands):
    parser = commands.add_parser("score", help="score a retained hex forecast against observations through Rust")
    parser.add_argument("run", type=Path, help="folder containing forecast history streams and receipt")
    parser.add_argument("--config", type=Path, help="observation/reference metadata overrides")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--budget", type=int, default=1800)
    parser.add_argument("--leads", help="comma-separated whole forecast hours, default every retained frame")
    parser.set_defaults(handler=command_main)


def command_main(args):
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else None
    card = score(args.run, config=config, out=args.out, timeout=args.timeout, budget=args.budget,
                 leads=set(map(int, args.leads.split(","))) if args.leads else None)
    print(json.dumps({"status": card["status"], "measured_metrics": card["measured_metrics"], "out": str(args.out or args.run / "observation-scorecard")}))
    return 0 if card["status"] == "MEASURED" else 3


def queue_retry(run, config):
    """A finite CPU-only waiter scores the run again after observations arrive."""
    root = Path(run) / "observation-scorecard"
    plan = root / "automatic-retry.json"
    atomic_json(plan, config)
    log = open(root / "automatic-retry.log", "ab")
    # CUDA_VISIBLE_DEVICES set and EMPTY is the contract the card-lease runners read (the GPU pool runner,
    # release_for_cpu_only_tail): once the forecast command exits and only this waiter is left, they give the
    # card back instead of holding the lease for CPU/network-only scoring (U-H7). Never unset it here.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", RAYON_NUM_THREADS="2")
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
    try:
        process = subprocess.Popen([sys.executable, "-m", "woof.hex.observation_scorecard", "watch", str(run), str(plan)],
                                   stdout=log, stderr=log, stdin=subprocess.DEVNULL, env=env, **kwargs)
        atomic_json(root / "automatic-retry-receipt.json", {"pid": process.pid, "config": str(plan), "status": "QUEUED"})
    finally:
        log.close()


def watch(run, plan):
    from woof.verification_visuals import _run_lock
    config = json.loads(Path(plan).read_text(encoding="utf-8"))
    root = Path(run) / "observation-scorecard"
    # OS ownership releases on process death. A repeated forecast hook cannot
    # leave two waiters scoring the same output.
    with _run_lock(root, name=".automatic-retry.lock"):
        _, frames = discover(run, config)
        delay = timedelta(hours=float(config.get("observation_delay_hours", 7)))
        start = instant(config["start"])
        # Frames valid in the future are not waited for (future_frames): their rows say when to score them.
        future = future_frames(frames, config)
        remaining = {int((instant(f["valid_time"]) - start).total_seconds() // 3600): instant(f["valid_time"]) + delay
                     for f in frames if f["valid_time"] not in future}
        card_path = root / "scorecard.json"
        card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.is_file() else {"status": "NOT_MEASURED"}
        if future:
            print(f"not waiting for {len(future)} frame(s) valid in the future; observable from "
                  f"{min(future.values())}: woof hex score {run}", flush=True)
        while remaining:
            now = datetime.now(timezone.utc)
            ready = {lead for lead, when in remaining.items() if when <= now}
            if not ready:
                time.sleep(min(60, max(1, (min(remaining.values()) - now).total_seconds())))
                continue
            card = score(run, config=config, budget=float(config.get("acquisition_budget_seconds", 3600)), leads=ready)
            for lead in ready:
                remaining.pop(lead)
        ready_at = instant(frames[-1]["valid_time"]) + delay
        receipt = {"status": card["status"], "completed_after": stamp(ready_at)}
        if future:
            receipt.update(observable_from=future, reason=f"{len(future)} frame(s) valid in the future were not waited "
                           f"for; score them from {min(future.values())} with: woof hex score {run}")
        atomic_json(root / "automatic-retry-receipt.json", receipt)
        released = []
        for key in ("owned_init_pin", "owned_keep_marker"):
            path = Path(config[key]) if config.get(key) else None
            if path is not None and path.exists() and path.resolve().is_relative_to(Path(run).resolve()):
                released.append({"path": str(path), "bytes": path.lstat().st_size, "kind": key})
                path.unlink()
        atomic_json(root / "released-retention.json", {"deleted_files": released})
    return 0 if card["status"] == "MEASURED" else 3


def _alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        kernel = ctypes.windll.kernel32
        handle = kernel.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        try:
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def completed_scorecard(run, config_path):
    """Reuse a measured card only for the same configuration, binary and forecast bytes.

    Missing observation coverage stays explicit in the card. It does not turn an otherwise
    completed global run into a process failure. The standalone score command keeps its
    coverage exit status and remains the explicit way to request fresh observations.
    """
    path = Path(run) / "observation-scorecard/scorecard.json"
    try:
        card = json.loads(path.read_text(encoding="utf-8"))
        if (card.get("error") or card.get("status") not in ("MEASURED", "PARTIALLY_MEASURED")
                or not card.get("measured_metrics", 0)):
            return None
        config, frames = discover(run, json.loads(Path(config_path).read_text(encoding="utf-8")))
        if card.get("configuration") != config:
            return None
        def digest(filename):
            with open(filename, "rb") as handle:
                return hashlib.file_digest(handle, "sha256").hexdigest()
        if card.get("native_binary_sha256") != digest(binary("hex_obs_scorecard")):
            return None
        recorded = {row[key]["path"]: row[key]["sha256"]
                    for row in card["sources"] for key in ("history", "diagnostics", "init") if key in row}
        required = {config["init"], *(f[key] for f in frames for key in ("history", "diagnostics") if key in f)}
        if any(filename not in recorded or digest(filename) != recorded[filename] for filename in required):
            return None
        if {f["valid_time"] for f in frames} - {r["valid_time"] for r in card["metrics"]}:
            return None
        return {"joined": "completed", "returncode": 0, "status": card["status"],
                "measured_metrics": card["measured_metrics"], "path": str(path),
                "note": f"scored once: {card['status']}, {card['measured_metrics']} measured metrics in {path}"}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def door_scorecard(run, config_path):
    """Every composed door hands scoring to a CPU worker, including older forecasts.

    Some forecast producers do not run forecast_hook. Falling back to the
    synchronous score command kept the whole door waiting for archive I/O.
    """
    joined = join_automatic_scorer(run, config_path) or completed_scorecard(run, config_path)
    if joined is not None:
        return joined
    if str(os.environ.get(SCORING_ENV, "")).strip().lower() == "off":
        # Nothing to discover for a run that asked for no scoring: the record says SKIPPED and why.
        scheduled = forecast_hook(run, init=None, grid=None, start=None, config_path=config_path)
    else:
        config, _ = discover(run, json.loads(Path(config_path).read_text(encoding="utf-8")))
        scheduled = forecast_hook(run, init=config["init"], grid=config["grid"], start=config["start"],
                                  config_path=config_path)
    if scheduled["status"] == "SKIPPED":
        # Scoring turned off (WOOF_HEX_OBSERVATION_SCORING=off, a door's --skip score) is an outcome, not a
        # failure. THE BREAKAGE THIS PREVENTS (both-routes matrix, woof hex 0.4.14 candidate, 2026-11-06): a
        # country door run with the variable set and its score stage still listed raised here after a finished
        # forecast, and the door exited 1.
        return {"joined": "skipped", **scheduled,
                "note": f"observation scoring SKIPPED ({SCORING_ENV}=off): {scheduled['path']}"}
    if scheduled["status"] in {"QUEUED", "NOT_YET_OBSERVABLE", "DEFERRED"}:
        return {"joined": "automatic", **scheduled,
                "note": f"CPU observation scoring {scheduled['status']}: {scheduled['path']}"}
    raise RuntimeError(f"could not schedule CPU observation scoring: {scheduled}")


def join_automatic_scorer(run, config_path, *, poll_seconds=10.0):
    """Join the forecast's automatic scorer when it was queued with this same config file.

    Returns None when the composed door must queue a CPU scorer: no automatic scorer, a different config
    (an explicit rescore), or its waiter died. A finished scorer's result is returned; a running one is left to its work and
    reported at once. Scoring is CPU work, so a door holding a card never waits for it: the door ran ~15 min
    on its card lease for a past case's scoring (2026-10-24). ``poll_seconds`` is kept for callers."""
    root = Path(run) / "observation-scorecard"
    source, receipt, plan = root / SOURCE_CONFIG, root / "automatic-retry-receipt.json", root / "automatic-retry.json"
    config_path = Path(config_path)
    if not (source.is_file() and receipt.is_file() and config_path.is_file()):
        return None
    if json.loads(source.read_text(encoding="utf-8")).get("sha256") != hashlib.sha256(config_path.read_bytes()).hexdigest():
        return None
    card_path = root / "scorecard.json"
    state = json.loads(receipt.read_text(encoding="utf-8"))
    if state.get("status") != "QUEUED":
        card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.is_file() else {}
        status = card.get("status", state.get("status"))
        return {"joined": "automatic", "status": status, "path": str(card_path),
                "note": f"scored once, by the forecast's automatic scorer: {status}, "
                        f"{card.get('measured_metrics', 0)} measured metrics in {card_path}"}
    if not plan.is_file() or not _alive(state.get("pid")):
        return None
    config = json.loads(plan.read_text(encoding="utf-8"))
    _, frames = discover(run, config)
    last = instant(frames[-1]["valid_time"]) + timedelta(hours=float(config.get("observation_delay_hours", 7)))
    when = (f"each lead as its observations arrive, the last after {stamp(last)}" if last > datetime.now(timezone.utc)
            else "now, every lead's observations being in")
    return {"joined": "automatic", "status": "QUEUED", "pid": state.get("pid"), "path": str(card_path),
            "note": f"the forecast's automatic scorer (pid {state.get('pid')}) owns this run and scores it on the "
                    f"CPU, {when}; no card is held for it: {card_path}"}


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "watch":
        raise SystemExit(watch(sys.argv[2], sys.argv[3]))
    raise SystemExit("Use woof hex score RUN/fc")
