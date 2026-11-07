"""Initial water state for woof hydro: orchestration only.

Rust does every data step (rw_zarr reads and crops the source, rw_hydro_init
builds area weights, maps the state onto catchments and runs the spin-up).
This module picks the table row, splits the fetch into time slices that run
as parallel rw_zarr processes, and chains the Rust stages. Sources are rows
in woof/data/hydro/init-state-sources.v1.json; adding one is table work.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time

TABLE = Path(__file__).resolve().parent.parent / "data" / "hydro" / "init-state-sources.v1.json"
INIT_ENV = "WOOF_RW_HYDRO_INIT"
ISO = "%Y-%m-%dT%H:%M:%SZ"


def load_table(path: Path | None = None) -> dict:
    return json.loads(Path(path or TABLE).read_text(encoding="utf-8"))


def admit(table: dict, name: str) -> dict:
    """The row, or a refusal naming why it cannot be used.

    Prevents: building a public product from a source whose licence was not
    read at the source, that needs an account the owner has not ruled on, or that
    has no reader yet (the row says which).
    """
    rows = table["rows"]
    if name not in rows:
        raise KeyError(f"no initial-state source row {name!r}; rows: {sorted(rows)}")
    row = rows[name]
    if row.get("status") != "admitted":
        raise PermissionError(f"initial-state source {name} is {row.get('status')}: {row.get('reason')}")
    if row.get("licence", {}).get("status") != "verified":
        raise PermissionError(f"initial-state source {name} has no licence read at the source")
    return row


def parse_time(text: str) -> datetime:
    return datetime.strptime(text, ISO).replace(tzinfo=timezone.utc)


def hours(start: datetime, count: int, step_s: int = 3600) -> list[str]:
    return [(start + timedelta(seconds=step_s * i)).strftime(ISO) for i in range(count)]


def _rw_zarr() -> Path:
    from woof.zarr_bridge import resolve_zarr_bin
    return resolve_zarr_bin()


def _rw_hydro_init() -> Path:
    override = os.environ.get(INIT_ENV)
    if override:
        return Path(override)
    # The bridge ladder (checkout build, wheel libexec, staged bundle), then
    # the on-demand stage a wheel install uses (woof.bridge_assets).
    from .door import _executable
    found = _executable({"env": INIT_ENV, "filename": "rw_hydro_init",
                         "crate": "tools/hydro_world_init"})
    if found is not None:
        return found
    raise FileNotFoundError(f"rw_hydro_init is not built; set {INIT_ENV} or build tools/hydro_world_init")


def _state_variables(row: dict) -> list[tuple[str, str]]:
    out = []
    state = row["state"]
    for group in ("soil_water", "soil_temperature"):
        for layer in state[group]["layers"]:
            out.append((layer["variable"], state[group]["units"]))
    for key in ("snow_water_equivalent", "snow_density", "snow_temperature", "land_fraction", "soil_class"):
        if key in state and "variable" in state[key]:
            out.append((state[key]["variable"], state[key]["units"]))
    return out


def _forcing_variables(row: dict) -> list[tuple[str, str]]:
    seen = {}
    for binding in row["forcing"].values():
        for term in binding["terms"]:
            seen[term["variable"]] = binding["units"]
    return list(seen.items())


def _zarr_request(row: dict, variables, times, area) -> dict:
    access = row["access"]
    return {"store": access["store"], "times": times, "area": list(area),
            "expected_levels_hpa": access["expected_levels_hpa"],
            "fields": [{"source": v, "output": v, "units": u, "pressure": False} for v, u in variables]}


def _fetch_jobs(row, area, state_times, forcing_times, slice_hours=24):
    jobs = []
    if state_times:
        jobs.append(("state", _zarr_request(row, _state_variables(row), state_times, area)))
    forcing_vars = _forcing_variables(row)
    for i in range(0, len(forcing_times), slice_hours):
        part = forcing_times[i:i + slice_hours]
        jobs.append((f"forcing-{i // slice_hours:04d}", _zarr_request(row, forcing_vars, part, area)))
    return jobs


def _payload_identity(path):
    """Hash opaque source bytes; scientific decoding remains in Rust."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"size_bytes": size, "sha256": digest.hexdigest()}


def _payload_matches(item):
    path, part = item
    if "size_bytes" not in part and "sha256" not in part:
        return True  # Legacy receipts carry no payload integrity evidence.
    size, digest = part.get("size_bytes"), part.get("sha256")
    if (type(size) is not int or size < 0 or not isinstance(digest, str)
            or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
            or path.stat().st_size != size):
        return False
    return _payload_identity(path) == {"size_bytes": size, "sha256": digest}


def _matching_fetch(directory, jobs):
    """Match the actual extraction requests, including crop, variables and units.

    New receipts also seal the opaque payload bytes. Legacy receipts remain
    usable, without claiming integrity evidence they did not record.
    """
    try:
        receipt = json.loads((directory / "fetch-receipt.json").read_text())
        parts = {part["name"]: part for part in receipt["slices"]}
        if len(parts) != len(receipt["slices"]) or set(parts) != {name for name, _ in jobs}:
            return None
        payloads = []
        for name, request in jobs:
            if (not (directory / f"{name}.nc").is_file()
                    or json.loads((directory / f"{name}.request.json").read_text()) != request):
                return None
            payloads.append((directory / f"{name}.nc", parts[name]))
        if payloads:
            with ThreadPoolExecutor(max_workers=min(len(payloads), os.cpu_count() or 1)) as pool:
                if not all(pool.map(_payload_matches, payloads)):
                    return None
        return receipt
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _fetch_location(directory, jobs):
    """Keep existing inputs and select a reusable directory for this request."""
    signature = hashlib.sha256(json.dumps(jobs, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()
    candidate, attempt = directory, 0
    while candidate.exists():
        receipt = _matching_fetch(candidate, jobs)
        if receipt is not None:
            return candidate, receipt
        # A different crop or incomplete attempt stays untouched. Later calls
        # find the same completed request instead of downloading it again.
        suffix = signature if attempt == 0 else f"{signature}-{attempt}"
        candidate = directory.with_name(f"{directory.name}-{suffix}")
        attempt += 1
    return candidate, None


def _shared_fetch(directory, row, *, jobs, area, state_times, forcing_times, progress):
    """One live writer per shared cache address; other preparations reuse it.

    Selection and extraction share the lease: observing another writer's
    partial directory must not start a second download of the same request.
    The OS releases ownership on interruption; incomplete inputs stay intact
    and the next preparation selects a fresh attempt.
    """
    from woof.remote_artifact_cache import Lease

    directory = Path(directory).resolve()
    selected, fetched = _fetch_location(directory, jobs)
    if fetched is not None:
        # Completed cache entries are immutable. Reuse remains read-only,
        # including a source extraction supplied by another preparation.
        progress(f"hydro init: reusing fetched source in {selected}")
        return selected, fetched
    locks = directory.parent / ".hydro-init-fetch-locks"
    locks.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(os.path.normcase(str(directory)).encode()).hexdigest()
    lock_path = locks / (key + ".lock")
    lease = Lease(lock_path)
    if lease.file is None:
        progress(f"hydro init: waiting for shared source in {directory}")
        lease = Lease(lock_path, timeout=float("inf"))
    with lease:
        selected, fetched = _fetch_location(directory, jobs)
        if fetched is None:
            fetched = fetch(row, area=area, state_times=state_times,
                            forcing_times=forcing_times, out=selected, progress=progress)
        else:
            progress(f"hydro init: reusing fetched source in {selected}")
        return selected, fetched


def fetch(row: dict, *, area, state_times, forcing_times, out: Path,
          workers: int | None = None, slice_hours: int = 24, progress=print) -> dict:
    """Fetch the state snapshots and the forcing series as parallel rw_zarr slices.

    Each rw_zarr process reads its slice's chunks with its own worker pool, so
    the fetch runs as many processes as there are slices (bounded by workers).
    """
    if row["access"]["protocol"] != "zarr-http" or row["access"]["reader"] != "rw_zarr":
        raise NotImplementedError(f"no fetch path for access {row['access']}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    exe = _rw_zarr()
    jobs = _fetch_jobs(row, area, state_times, forcing_times, slice_hours)
    workers = workers or min(len(jobs), max(1, (os.cpu_count() or 8)))
    started = time.perf_counter()

    def run(job):
        name, request = job
        req = out / f"{name}.request.json"
        req.write_text(json.dumps(request, indent=1), encoding="utf-8")
        t0 = time.perf_counter()
        proc = subprocess.run([os.fspath(exe), "extract", os.fspath(req), os.fspath(out / f"{name}.nc")],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"rw_zarr {name} failed: {proc.stderr[-2000:]}")
        identity = _payload_identity(out / f"{name}.nc")
        return {"name": name, "seconds": time.perf_counter() - t0, "times": len(request["times"]),
                "fields": len(request["fields"]), "file": f"{name}.nc", **identity}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run, jobs))
    receipt = {"schema": "arwen.hydro-world.init-fetch.v1", "store": row["access"]["store"],
               "area_south_west_north_east": list(area), "state_times": state_times,
               "forcing_first": forcing_times[0] if forcing_times else None,
               "forcing_last": forcing_times[-1] if forcing_times else None,
               "forcing_records": len(forcing_times), "slices": results,
               "wall_seconds": time.perf_counter() - started,
               "parallelism": f"{workers} rw_zarr processes, each reading its fields concurrently",
               "chunks_read": sum(r["times"] * r["fields"] for r in results)}
    (out / "fetch-receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    progress(f"hydro init: fetched {receipt['chunks_read']} source chunks in {receipt['wall_seconds']:.1f} s")
    return receipt


#: The live analysis row a forecast past the reanalysis's last hour takes its
#: soil water from where it covers the domain (coordinator decision, 2026-10-21).
LIVE_ROW = "nwm-analysis-assim-conus"


#: Each store's last coverage answer (``rw_zarr coverage``), kept so the
#: receipt can say which spin-up hours were finalized and which preliminary.
_COVERAGE: dict[str, dict] = {}
#: (monotonic seconds, last whole hour) per store, reused for COVERAGE_REUSE_S.
_COVERAGE_ASKED: dict[str, tuple[float, datetime]] = {}
COVERAGE_REUSE_S = 600.0


def source_last_time(row: dict) -> datetime | None:
    """The last whole hour the row's store serves (``rw_zarr coverage``).

    None when the row has no Zarr store or the store cannot be asked; the
    plan is then the one this module always made, and a fetch past the end
    fails with rw_zarr's own refusal.
    """
    access = row.get("access", {})
    if access.get("reader") != "rw_zarr" or not access.get("store"):
        return None
    # The source prefetch (prefetch) and the preparation it feeds ask the same
    # store within seconds of each other: one answer per store per process for
    # a few minutes, not a second rw_zarr coverage round trip.
    asked = _COVERAGE_ASKED.get(access["store"])
    if asked is not None and time.monotonic() - asked[0] < COVERAGE_REUSE_S:
        return asked[1]
    try:
        proc = subprocess.run([os.fspath(_rw_zarr()), "coverage", access["store"]],
                              capture_output=True, text=True, timeout=180)
        if proc.returncode != 0:
            return None
        coverage = json.loads(proc.stdout)
        stop = parse_time(coverage["accepted_stop"])
    except Exception:  # noqa: BLE001 - an unanswered question keeps the old plan
        return None
    _COVERAGE[access["store"]] = coverage
    last = stop.replace(minute=0, second=0, microsecond=0)
    _COVERAGE_ASKED[access["store"]] = (time.monotonic(), last)
    return last


def _download(url: str, path: Path) -> dict:
    """Opaque bytes to a file (decoding stays in Rust); size and SHA-256."""
    import urllib.request
    part = path.with_name(path.name + ".part")
    digest, size = hashlib.sha256(), 0
    with urllib.request.urlopen(url, timeout=120) as response, part.open("wb") as stream:
        for block in iter(lambda: response.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
            stream.write(block)
    part.replace(path)
    return {"url": url, "size_bytes": size, "sha256": digest.hexdigest()}


def live_soil(*, fabric, moment: datetime, area, out: Path, table: dict, progress=print):
    """Soil water at the forecast start from the live analysis row, mapped onto
    the catchments it covers (the others NaN). Returns (state_dir, receipt), or
    raises naming why this domain or time has none.
    """
    row = admit(table, LIVE_ROW)
    access = row["access"]
    west, south, east, north = access["domain_bbox_west_south_east_north"]
    a_south, a_west, a_north, a_east = area
    if a_east < west or a_west > east or a_north < south or a_south > north:
        raise LookupError(f"{LIVE_ROW} covers {west}..{east} E, {south}..{north} N; this domain "
                          f"({a_west:.2f}..{a_east:.2f} E, {a_south:.2f}..{a_north:.2f} N) lies outside it")
    hour = moment.replace(minute=0, second=0, microsecond=0)
    age_days = (datetime.now(timezone.utc) - hour).total_seconds() / 86400.0
    if age_days > access["retention_days"]:
        raise LookupError(f"{LIVE_ROW} keeps {access['retention_days']} days; "
                          f"{hour:%Y-%m-%dT%H}Z is {age_days:.1f} days old")
    url = access["url_template"].format(yyyymmdd=hour.strftime("%Y%m%d"), hh=hour.strftime("%H"))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    land = out / "land.nc"
    started = time.perf_counter()
    fetched = _download(url, land)
    fetched["seconds"] = time.perf_counter() - started
    progress(f"hydro init: live analysis {url} ({fetched['size_bytes'] / 1e6:.0f} MB, {fetched['seconds']:.1f} s)")
    weights = run_stage("weights", {"fabric": os.fspath(fabric), "grid_file": os.fspath(land),
                                    "row": LIVE_ROW, "table": os.fspath(TABLE),
                                    "output": os.fspath(out / "weights")}, out / "weights-stage")
    state = run_stage("state", {"table": os.fspath(TABLE), "row": LIVE_ROW, "fabric": os.fspath(fabric),
                                "weights": os.fspath(out / "weights"), "source_file": os.fspath(land),
                                "valid_time": hour.strftime(ISO), "partial_coverage": True,
                                "output": os.fspath(out / "state")}, out / "state-stage")
    counts = state.get("counts", {})
    if counts.get("catchments_not_covered_by_source") == counts.get("catchments"):
        raise LookupError(f"{LIVE_ROW} covers none of this fabric's catchments")
    return out / "state", {"row": LIVE_ROW, "valid_time": hour.strftime(ISO), "fetch": fetched,
                           "weights": weights, "state": state}


def _days_root():
    """The forcing-day cache: beside the fabric cache, shared by every domain and start."""
    from .door import cache_root
    return cache_root() / "forcing-days"


def _split_days(times):
    """(whole UTC days {day: 24 hourly times}, the remaining edge times)."""
    by_day = {}
    for stamp in times:
        by_day.setdefault(stamp[:10], []).append(stamp)
    full = {day: stamps for day, stamps in by_day.items() if len(stamps) == 24}
    edge = [stamp for stamp in times if stamp[:10] not in full]
    return full, edge


def forcing_days(row, area, days, root, *, workers=None, progress=print):
    """One extract per whole UTC day, kept in ``root`` and shared by every
    start on the same area, store and variables; missing days are fetched in
    parallel (one rw_zarr process each, every process reading its fields
    concurrently). Returns (files in time order, receipt)."""
    if not days:
        return [], {"days": 0}
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    plan = []
    for day in sorted(days):
        request = _zarr_request(row, _forcing_variables(row), days[day], area)
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:24]
        plan.append((day, request, root / f"{day}-{key}.nc"))
    missing = [item for item in plan if not (item[2].is_file() and item[2].stat().st_size > 0)]
    exe = _rw_zarr() if missing else None

    def run(item):
        day, request, path = item
        part = path.with_name(f"{path.name}.part.{os.getpid()}.{threading.get_ident()}")
        req = path.with_name(path.name + ".request.json")
        req.write_text(json.dumps(request, indent=1), encoding="utf-8")
        proc = subprocess.run([os.fspath(exe), "extract", os.fspath(req), os.fspath(part)],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"rw_zarr day {day} failed: {proc.stderr[-2000:]}")
        os.replace(part, path)

    if missing:
        workers = workers or min(len(missing), max(1, os.cpu_count() or 8))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run, missing))
    receipt = {"days": len(plan), "fetched_days": len(missing), "reused_days": len(plan) - len(missing),
               "cache": os.fspath(root), "seconds": time.perf_counter() - started,
               "parallelism": f"{min(len(missing), os.cpu_count() or 8) if missing else 0} rw_zarr processes"}
    progress(f"hydro init: spin-up forcing days {receipt['reused_days']} reused, {receipt['fetched_days']} "
             f"fetched in {receipt['seconds']:.1f} s ({root})")
    return [os.fspath(item[2]) for item in plan], receipt


#: How many times a spin-up whose source has a hole no fill source can close
#: is moved earlier (ending before the hole, starting earlier by as much, so
#: it keeps its whole length) before the preparation fails by name.
MAX_HOLD_BACKS = 6


def _fill_days_root(name):
    """Fill-source day files: beside the forcing-day cache, shared by every domain."""
    return _days_root().parent / "fill-days" / name


def fill_gaps(row, files, variables, out, *, progress=print):
    """Close the holes of the fetched source files from the row's next source.

    rw_hydro_init forcing-gaps lists every (file, record, variable) holding a
    non-finite value; the row's ``gap_fill`` source's day files for those UTC
    days are downloaded in parallel (opaque bytes, cached and shared); and
    rw_hydro_init forcing-fill writes a patched copy of each holed file.
    Returns ({file: patched copy}, receipt, [times still missing]).
    """
    out = Path(out)
    found = run_stage("forcing-gaps", {"files": [os.fspath(f) for f in files], "variables": list(variables)},
                      out / "gaps-stage").get("gaps", [])
    if not found:
        return {}, {"gaps": 0}, []
    fill = row.get("gap_fill")
    need = sorted({(gap["time"][:10], var) for gap in found for var in gap["variables"]})
    paths, failed = {}, {}
    if fill:
        root = _fill_days_root(fill["name"])
        root.mkdir(parents=True, exist_ok=True)

        def get(item):
            day, var = item
            path = root / f"{day}-{var}.nc"
            if path.is_file() and path.stat().st_size > 0:
                return item, path, None
            url = fill["url_template"].format(yyyy=day[:4], mm=day[5:7], dd=day[8:10], variable=var)
            try:
                _download(url, path)
            except Exception as error:  # noqa: BLE001 - the hole stays and is named
                return item, None, f"{url}: {type(error).__name__}: {error}"
            return item, path, None

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=min(len(need), 16)) as pool:
            for item, path, error in pool.map(get, need):
                if path is None:
                    failed[item] = error
                else:
                    paths[item] = path
        progress(f"hydro init: {len(found)} source record(s) hold NaN; {len(paths)} {fill['name']} day file(s) "
                 f"ready, {len(failed)} unavailable ({time.perf_counter() - started:.1f} s, parallel downloads)")
    by_file = {}
    for gap in found:
        by_file.setdefault(gap["file"], []).append(gap)
    fills, missing = [], set()
    for index, (file, gaps) in enumerate(sorted(by_file.items())):
        records = []
        for gap in gaps:
            sources = [{"variable": var, "path": os.fspath(paths[(gap["time"][:10], var)])}
                       for var in gap["variables"] if (gap["time"][:10], var) in paths]
            missing.update(gap["time"] for var in gap["variables"] if (gap["time"][:10], var) not in paths)
            if sources:
                records.append({"time": gap["time"], "sources": sources})
        if records:
            fills.append({"file": file, "records": records,
                          "output": os.fspath(out / f"{index:04d}-{Path(file).stem}-filled.nc")})
    replaced, patched = {}, []
    if fills:
        out.mkdir(parents=True, exist_ok=True)
        done = run_stage("forcing-fill", {"fills": fills}, out / "fill-stage")
        for item in done.get("files", []):
            replaced[item["file"]] = item["output"]
            patched.extend(item.get("patched", []))
            missing.update(rec["time"] for rec in item.get("still_missing", []))
    receipt = {"gaps": len(found), "fill_source": fill["name"] if fill else None,
               "first_gap": min(g["time"] for g in found), "last_gap": max(g["time"] for g in found),
               "records_patched": len({(p["time"], p["variable"]) for p in patched}),
               "patched_times": sorted({p["time"] for p in patched}),
               "unavailable": {f"{day} {var}": error for (day, var), error in failed.items()},
               "still_missing": sorted(missing)}
    return replaced, receipt, sorted(missing)


def spinup_segments(start, end, *, source, finalized, patched_times, fill_name):
    """The spin-up's forcing hours (stamps start+1 h .. end) as runs of one source each."""
    patched = set(patched_times)
    runs, hour = [], start + timedelta(hours=1)
    while hour <= end:
        stamp = hour.strftime(ISO)
        if stamp in patched:
            label = f"{fill_name} (filled: the {source} store served NaN)"
        elif finalized is None:
            label = source
        elif hour <= finalized:
            label = f"{source} finalized reanalysis"
        else:
            label = f"{source} preliminary (ERA5T)"
        if runs and runs[-1]["source"] == label:
            runs[-1]["last"], runs[-1]["hours"] = stamp, runs[-1]["hours"] + 1
        else:
            runs.append({"first": stamp, "last": stamp, "hours": 1, "source": label})
        hour += timedelta(hours=1)
    return runs


def _fetch_window(row, area, state_times, forcing_times, fetch_dir, *, progress=print):
    """The state snapshots plus edge hours (this request's own) and the whole
    forcing days (the shared day cache), fetched AT THE SAME TIME.

    Breakage it prevents (2.9.4 R11, 2026-10-22): the 89 forcing days waited
    for the state and edge extraction to finish (11-14 s of a first run's
    hydro preparation) although the two read different records of the store
    into different files. Returns (fetch dir, fetch receipt, day files, day receipt).
    """
    full_days, edge_times = _split_days(forcing_times)
    jobs = _fetch_jobs(row, area, state_times, edge_times)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="hydro-forcing-days") as pool:
        days = (pool.submit(forcing_days, row, area, full_days, _days_root(), progress=progress)
                if full_days else None)
        selected, fetched = _shared_fetch(fetch_dir, row, jobs=jobs, area=area,
                                          state_times=state_times, forcing_times=edge_times,
                                          progress=progress)
        day_files, day_receipt = days.result() if days is not None else ([], {"days": 0})
    return selected, fetched, day_files, day_receipt


def _area_contains(outer, inner) -> bool:
    """Whether the [south, west, north, east] box ``outer`` holds ``inner``."""
    return (outer[0] <= inner[0] and outer[1] <= inner[1]
            and outer[2] >= inner[2] and outer[3] >= inner[3])


#: How far the source area a launch derives from its domain footprint reaches
#: past it: the fabric's own box pad (``bbox`` stage, 0.5 degree) plus a
#: quarter degree (one ERA5 cell) for catchments drawn a little past the footprint.
FOOTPRINT_AREA_PAD_DEGREES = 0.75


def footprint_area(footprint, pad=FOOTPRINT_AREA_PAD_DEGREES):
    """The [south, west, north, east] source area for a fabric over ``footprint``.

    Known before the fabric exists, so the source fetch can start beside the
    fabric's own source downloads and build (:func:`prefetch`); the
    preparation then uses it whenever it holds the built fabric's box
    (:func:`prepare` ``area_hint``). None when the footprint has no box.
    """
    try:
        south, west = float(footprint["south"]), float(footprint["west"])
        north, east = float(footprint["north"]), float(footprint["east"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (south < north and west < east):
        return None
    return [south - pad, west - pad, north + pad, east + pad]


def _source_window(row, date, spinup_days):
    """(forecast start, last spin-up hour) for ``date``: the store's last hour
    when the forecast lies past it, as :func:`_prepare` plans it."""
    t = parse_time(date)
    last = source_last_time(row) if spinup_days else None
    return t, (last if (last is not None and t > last) else t)


def _fetch_dir_for(requested_out):
    return Path(requested_out).absolute().with_name(Path(requested_out).name + "-source")


def prefetch(*, date: str, out: Path, area, source: str = "era5-arco", spinup_days: int = 0,
             progress=print) -> dict:
    """Fetch the source a later :func:`prepare` with the same ``out``, ``date``
    and area will read: the state snapshots, the edge hours and the whole
    forcing days, into the same caches, so that preparation reuses them.

    For a launch whose fabric is still being built: the fetch does not read
    the fabric, only the area (:func:`footprint_area`). Breakage it prevents
    (2.9.4 R11, 2026-10-22): a first run on a new region spent 107-165 s
    extracting 89 ERA5 forcing days only after the fabric's own source
    downloads and build (45-160 s) had finished, all of it before step 1.
    A hole the store serves as NaN is left to :func:`prepare` (it fills it or
    moves the window). Returns the fetch receipts.
    """
    row = admit(load_table(), source)
    _, t_source = _source_window(row, date, spinup_days)
    start = t_source - timedelta(days=spinup_days)
    state_times = sorted({start.strftime(ISO), t_source.strftime(ISO)})
    forcing_times = hours(start + timedelta(hours=1), spinup_days * 24) if spinup_days else []
    started = time.perf_counter()
    selected, fetched, _files, days = _fetch_window(
        row, list(area), state_times, forcing_times, _fetch_dir_for(out), progress=progress)
    return {"fetch_dir": os.fspath(selected), "fetch": fetched, "forcing_days": days,
            "area_south_west_north_east": list(area), "seconds": time.perf_counter() - started}


def run_stage(command: str, request: dict, workdir: Path) -> dict:
    """One rw_hydro_init stage from a request file; returns its JSON receipt."""
    workdir.mkdir(parents=True, exist_ok=True)
    req = workdir / f"{command}.request.json"
    req.write_text(json.dumps(request, indent=1), encoding="utf-8")
    proc = subprocess.run([os.fspath(_rw_hydro_init()), command, os.fspath(req)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"rw_hydro_init {command} failed: {proc.stderr[-4000:]}")
    return json.loads(proc.stdout)


def prepare(*, fabric: Path, date: str, out: Path, source: str = "era5-arco",
            spinup_days: int = 0, area=None, params: Path | None = None,
            fetch_dir: Path | None = None, trace=(), progress=print, area_hint=None) -> dict:
    """Prepare fresh results, preserving previous runs and reusing source data.

    ``area_hint``: the source area a launch derived from its footprint
    (:func:`footprint_area`) and may already have fetched (:func:`prefetch`).
    With no ``area``, it is used whenever it holds the fabric's own box, so
    every preparation of one domain reads the same source files; otherwise the
    fabric's box is used, as without a hint.
    """
    from woof.output_claim import acquire_output
    from .output import _select_output_directory

    # Invalid source/date requests must fail before creating output or leases.
    admit(load_table(), source)
    parse_time(date)
    requested = Path(out).absolute()
    selected = _select_output_directory(requested)

    def reserve():
        selected.mkdir(parents=True, exist_ok=True)
        if selected.is_symlink() or next(selected.iterdir(), None) is not None:
            raise FileExistsError(f"initial-state output acquired other files: {selected}")
        return selected

    while True:
        try:
            claim = acquire_output(selected, prepare=reserve)
            break
        except FileExistsError:
            # Another preparation can reserve an empty directory between
            # selection and acquisition. A new sibling keeps both runs safe.
            selected = Path(tempfile.mkdtemp(prefix=requested.name + "-",
                                             dir=requested.absolute().parent))
    with claim:
        progress(f"Hydro initial state: {claim.path} (previous files preserved)")
        return _prepare(fabric=fabric, date=date, out=claim.path, source=source,
                        spinup_days=spinup_days, area=area, params=params,
                        fetch_dir=fetch_dir, trace=trace, progress=progress,
                        requested_out=requested, area_hint=area_hint)


def _prepare(*, fabric, date, out, source, spinup_days, area, params,
             fetch_dir, trace, progress, requested_out, area_hint=None):
    """Initial state for every catchment of FABRIC at DATE, optionally spun up.

    With spinup_days N the source state at DATE - N days is mapped onto the
    catchments and CFE runs N days of the source's own forcing to DATE.
    """
    import sys
    table = load_table()
    row = admit(table, source)
    # A forecast past the reanalysis's last hour (every live and real-time
    # run: ERA5 lags about six days) spins up to that hour and is carried to
    # the forecast start; where the live analysis covers the domain, its soil
    # water replaces the carried soil at the start. Hindcasts inside the
    # record keep the spin-up to the forecast start itself. Breakage it
    # prevents: every live run's hydro preparation failed on the ERA5 fetch
    # ("outside the coverage this store declares") and the forecast ran with
    # hydro off (fabric_preparation_failed).
    t, t_source = _source_window(row, date, spinup_days)
    out = Path(out)
    area_source = "given"
    if area is None:
        box = run_stage("bbox", {"fabric": os.fspath(fabric), "pad_degrees": 0.5}, out / "bbox")["area_south_west_north_east"]
        if area_hint is not None and _area_contains(area_hint, box):
            area, area_source = list(area_hint), "footprint"
        else:
            area, area_source = box, "fabric box"
            if area_hint is not None:
                progress(f"hydro init: the fabric's box {box} reaches past the footprint area "
                         f"{list(area_hint)}; the source is read over the fabric's box")
    if fetch_dir is None:
        # Source cache belongs to the requested address, independent of fresh
        # result directories. Existing pre-cache-layout extractions still work.
        first_start = t_source - timedelta(days=spinup_days)
        first_times = hours(first_start + timedelta(hours=1), spinup_days * 24) if spinup_days else []
        first_jobs = _fetch_jobs(row, area, sorted({first_start.strftime(ISO), t_source.strftime(ISO)}),
                                 _split_days(first_times)[1])
        legacy = requested_out / "source"
        if not requested_out.is_symlink() and _matching_fetch(legacy, first_jobs) is not None:
            fetch_dir = legacy
        else:
            fetch_dir = requested_out.with_name(requested_out.name + "-source")
    variables = [name for name, _ in _forcing_variables(row) + _state_variables(row)]
    # Records the source serves as NaN are filled from the row's next source
    # (fill_gaps). A hole nothing fills moves the whole window earlier: the
    # spin-up ends before it, starts earlier by as much (it keeps all its
    # hours) and is carried to the start, and says so. Breakage it prevents
    # (2.9.4 WOOF readiness, 2026-10-22): ERA5T days the store served as NaN
    # made the spin-up give up on starts inside the record (hydro off, after
    # up to 12 min of preparation) and run 1,992 of 2,160 h on carried ones.
    held_back = []
    for attempt in range(MAX_HOLD_BACKS + 1):
        start = t_source - timedelta(days=spinup_days)
        state_times = sorted({start.strftime(ISO), t_source.strftime(ISO)})
        forcing_times = hours(start + timedelta(hours=1), spinup_days * 24) if spinup_days else []
        # Whole UTC days of spin-up forcing come from a cache shared by every start
        # on this area (forcing_days); only the partial edge days are this
        # request's own. Breakage it prevents: every start re-fetched all 8,666
        # ERA5 chunks (about 300 s on the HRRR recipe, a 4090 host), because the 24 h
        # slices began at each start's own hour and never matched a cached one.
        selected, fetched, day_files, day_receipt = _fetch_window(
            row, area, state_times, forcing_times, fetch_dir, progress=progress)
        state_nc = os.fspath(selected / "state.nc")
        files = sorted(os.fspath(p) for p in selected.glob("forcing-*.nc")) + day_files
        if not spinup_days:
            replaced, filled = {}, {"gaps": 0}
            break
        replaced, filled, missing = fill_gaps(row, [state_nc] + files, variables,
                                              out / "gap-fill" / f"window-{attempt}", progress=progress)
        if not missing:
            break
        moved_to = parse_time(missing[0]) - timedelta(hours=1)
        held_back.append({"window_end": t_source.strftime(ISO), "unfilled_from": missing[0],
                          "unfilled_records": len(missing), "moved_end_to": moved_to.strftime(ISO),
                          "unavailable": filled.get("unavailable", {})})
        message = (f"hydro init: {source} has {len(missing)} record(s) from {missing[0]} that no fill source "
                   f"closes; the {spinup_days * 24} h spin-up moves earlier to end {moved_to.strftime(ISO)} "
                   f"(it keeps its whole length) and is carried to the forecast start {date}")
        progress(message)
        print(message, file=sys.stderr)
        t_source = moved_to
    else:
        raise RuntimeError(f"hydro init: the {source} spin-up still has unfilled records after moving "
                           f"{MAX_HOLD_BACKS} times earlier: {held_back}")
    carry = t_source < t
    fetch_dir = selected
    state_nc = replaced.get(state_nc, state_nc)
    files = [replaced.get(f, f) for f in files]

    weights = run_stage("weights", {"fabric": os.fspath(fabric), "grid_file": state_nc,
                                    "row": source, "table": os.fspath(TABLE),
                                    "output": os.fspath(out / "weights")}, out / "weights-stage")
    state_at = start.strftime(ISO)
    state = run_stage("state", {"table": os.fspath(TABLE), "row": source, "fabric": os.fspath(fabric),
                                "weights": os.fspath(out / "weights"), "source_file": state_nc,
                                "valid_time": state_at, "params": os.fspath(params) if params else None,
                                "output": os.fspath(out / "state")}, out / "state-stage")
    result = {"schema": "arwen.hydro-world.init-run.v1", "fabric": os.fspath(fabric), "date": date,
              "requested_output_directory": os.fspath(requested_out), "output_directory": os.fspath(out),
              "source": source, "spinup_days": spinup_days, "fetch": fetched, "forcing_days": day_receipt,
              "gap_fill": filled, "weights": weights, "state": state,
              "source_area": {"area_south_west_north_east": list(area), "from": area_source}}
    if spinup_days:
        # Reanalysis state at the forecast start itself (Noah-OWP soil, snow),
        # beside the spun-up CFE storages.
        result["state_at_date"] = run_stage("state", {"table": os.fspath(TABLE), "row": source, "fabric": os.fspath(fabric),
                                                      "weights": os.fspath(out / "weights"), "source_file": state_nc,
                                                      "valid_time": t_source.strftime(ISO), "params": os.fspath(params) if params else None,
                                                      "output": os.fspath(out / "state-at-date")}, out / "state-at-date-stage")
        spin = {"table": os.fspath(TABLE), "row": source, "fabric": os.fspath(fabric),
                "weights": os.fspath(out / "weights"),
                "initial_state": os.fspath(out / "state"),
                "land_file": state_nc,
                "forcing_files": files, "start": state_at, "hours": spinup_days * 24,
                "params": os.fspath(params) if params else None,
                "trace": list(trace),
                "retain_runoff_replay": True,
                "output": os.fspath(out / "spinup")}
        coverage = _COVERAGE.get(row.get("access", {}).get("store")) or {}
        finalized = (parse_time(coverage["finalized_stop"][:14] + "00:00Z")
                     if coverage.get("finalized_stop") else None)
        segments = spinup_segments(start, t_source, source=source, finalized=finalized,
                                   patched_times=filled.get("patched_times", ()),
                                   fill_name=(row.get("gap_fill") or {}).get("name"))
        initial_water = {"method": "reanalysis_spinup", "source": source, "spinup_end": t_source.strftime(ISO),
                         "spinup_hours": {"requested": spinup_days * 24, "ran": spinup_days * 24},
                         "segments": segments}
        if held_back:
            initial_water["held_back"] = held_back
        if carry:
            gap = (t - t_source).total_seconds() / 3600.0
            spin["carry_to"] = date
            spin["carry_reason"] = (
                f"{source} serves through {t_source.strftime(ISO)}; the forecast starts {gap:.0f} h later"
                if not held_back else
                f"{source} has records no fill source closes from {held_back[-1]['unfilled_from']}; the spin-up "
                f"ends {t_source.strftime(ISO)} and the forecast starts {gap:.0f} h later")
            try:
                live_dir, live = live_soil(fabric=fabric, moment=t, area=area, out=out / "live",
                                           table=table, progress=progress)
                spin["soil_override"] = os.fspath(live_dir)
                initial_water.update(method="live_analysis_soil_over_carried_spinup", live=live, carried_hours=gap)
                progress(f"hydro init: soil water from {LIVE_ROW} at {live['valid_time']}; groundwater and "
                         f"channels from the {source} spin-up to {t_source.strftime(ISO)}, carried {gap:.0f} h")
                carried_label = f"carried: spun-up groundwater and channels held, soil water from {LIVE_ROW}"
            except Exception as error:  # noqa: BLE001 - the named fallback keeps hydro on
                initial_water.update(method="era5_carry_over", fallback=True, carried_hours=gap,
                                     live_analysis_unavailable=f"{type(error).__name__}: {error}")
                message = (f"hydro init: FALLBACK era5_carry_over -- no live analysis for this run "
                           f"({type(error).__name__}: {error}); the {source} spin-up to "
                           f"{t_source.strftime(ISO)} is held {gap:.0f} h to the forecast start {date}")
                progress(message)
                print(message, file=sys.stderr)
                carried_label = "carried: the spun-up state held to the forecast start (no forcing)"
            segments.append({"first": (t_source + timedelta(hours=1)).strftime(ISO), "last": date,
                             "hours": int(round(gap)), "source": carried_label})
        progress("hydro init: spin-up segments: " + "; ".join(
            f"{seg['first']}..{seg['last']} {seg['hours']} h {seg['source']}" for seg in segments))
        result["spinup"] = run_stage("spinup", spin, out / "spinup-stage")
        carried = result["spinup"].get("carry_over") if isinstance(result["spinup"], dict) else None
        if carried:
            # The spin-up's own account: a preliminary stream can end its
            # complete days before the store's declared stop.
            initial_water.update(spinup_end=carried["spinup_end"], carried_hours=carried["hours"],
                                 spinup_truncated=carried.get("spinup_truncated"))
            truncated = carried.get("spinup_truncated")
            if truncated:
                initial_water["spinup_hours"]["ran"] = truncated.get("ran_hours")
                message = (f"hydro init: SHORT spin-up -- ran {truncated.get('ran_hours')} of "
                           f"{truncated.get('requested_hours')} h: {truncated.get('reason')}")
                progress(message)
                print(message, file=sys.stderr)
        result["initial_water"] = initial_water
    (out / "init-run.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def register_cli(commands):
    p = commands.add_parser("init-state", help="initial soil, snow, groundwater and channel state from a reanalysis row, with optional CFE spin-up")
    p.add_argument("--fabric", type=Path, required=True, help="HYDRO-WORLD contract GeoPackage")
    p.add_argument("--date", required=True, help="forecast start, e.g. 2023-01-01T00:00:00Z")
    p.add_argument("--source", default="era5-arco", help="row in init-state-sources.v1.json")
    p.add_argument("--spinup-days", type=int, default=0)
    p.add_argument("--params", type=Path, help="CFE parameter row file when the fabric has no model_attributes")
    p.add_argument("--outdir", type=Path, required=True)
    p.set_defaults(func=lambda a: (prepare(fabric=a.fabric, date=a.date, out=a.outdir, source=a.source,
                                           spinup_days=a.spinup_days, params=a.params), 0)[1])
