"""Acquire the open source inputs a hydro fabric provider row needs for one domain.

Owner, 2026-10-18: hydro is default on whenever a fabric covers the domain, and
worldwide. A default run on a fresh install has no source tiles, so the door
fetches exactly the tiles and archives the footprint needs into a shared cache
(``$GPUWM_HYDRO_WORLD_TILES``, default ``~/.woof/hydro-world``), then binds the
per-domain parameter recipe.

Orchestration only. The source tables name every URL, local layout and licence:
``tools/hydro_world/data/sources.v1.json`` (fabric builder: DEM tiles, lakes,
dams) and ``tools/hydro_world_params/data/params.v1.json`` (parameters: DEM,
SoilGrids, WorldCover). The DEM tile roster comes from the builder's own
``rw_hydro_world tiles`` answer. This module moves bytes (parallel network
I/O, archive unpacking) and never decodes a raster. Every source used here is
licence-verified in those tables; nothing NC-licensed is reachable.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

ENV_ROOT = "GPUWM_HYDRO_WORLD_TILES"
DEFAULT_ROOT = "~/.woof/hydro-world"
WORKERS = 16
# The ISRIC WCS answers a few large subsets at a time; six at once drew HTTP
# 504s on the 23 x 17 degree Milton box.
SOIL_WORKERS = 3
#: Transient answers (gateway timeouts, throttling, resets) retry with
#: exponential backoff and jitter; a tile that still fails is retried once more
#: after every other input has finished, and only then does acquisition fail.
TRANSIENT_HTTP = (408, 425, 429, 500, 502, 503, 504)
ATTEMPTS = 6
BACKOFF_CAP_SECONDS = 60.0
# SoilGrids WCS answers a 0.0025 degree subset up to about this many pixels per
# request (the Rhine proof fetched 3400 x 3200 = 10.9 M); larger domains take a
# coarser grid so every request stays a size the server serves.
SOIL_MAX_PIXELS = 11_000_000
SOIL_BASE_DEG = 0.0025
USER_AGENT = "gpuwm-hydro-acquire/1"


def repo_root():
    return Path(__file__).resolve().parents[2]


def source_table_path():
    return repo_root() / "tools/hydro_world/data/sources.v1.json"


def params_table_path():
    return repo_root() / "tools/hydro_world_params/data/params.v1.json"


def _fill_tile(pattern, lat, lon):
    """Same tile naming as hydro_world_params::table::fill_tile."""
    return (pattern.replace("{ns}", "N" if lat >= 0 else "S")
            .replace("{lat2}", f"{abs(lat):02d}")
            .replace("{ew}", "E" if lon >= 0 else "W")
            .replace("{lon3}", f"{abs(lon):03d}"))


def _lon_ranges(west, east):
    """Longitude intervals of a box; a dateline box splits in two."""
    return [(west, east)] if west <= east else [(west, 180.0), (-180.0, east)]


def _tiles(south, north, west, east, step):
    out = []
    lat = math.floor(south / step) * step
    while lat < north:
        for w, e in _lon_ranges(west, east):
            lon = math.floor(w / step) * step
            while lon < e:
                out.append((int(lat), int(lon)))
                lon += step
        lat += step
    return out


def _download(url, path, *, absent_ok=False, timeout=900, attempts=ATTEMPTS):
    """Fetch one URL to ``path`` unless present. Returns present/fetched/absent.

    ``absent_ok`` is for tiled sources that publish no tile over open sea (the
    Copernicus DEM and WorldCover buckets): a 403/404 is listed in the tile
    directory's ``ABSENT.txt``, the list ``rw_hydro_world build`` reads to tell
    sea from a missing download (tools/hydro_world/hydro_world_build.py writes
    the same list), so later runs do not ask again.
    """
    path = Path(path)
    if path.is_file() and path.stat().st_size > 0:
        return "present", 0
    if absent_ok and path.name in _absent(path.parent):
        return "absent", 0
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.name}.part.{os.getpid()}.{threading.get_ident()}")
    error = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response, part.open("wb") as out:
                shutil.copyfileobj(response, out, 8 * 1024 * 1024)
            size = part.stat().st_size
            if size == 0:
                raise OSError(f"empty response from {url}")
            os.replace(part, path)
            return "fetched", size
        except urllib.error.HTTPError as http:
            part.unlink(missing_ok=True)
            if absent_ok and http.code in (403, 404):
                _mark_absent(path)
                return "absent", 0
            error = http
            if http.code not in TRANSIENT_HTTP:
                break
        except (OSError, urllib.error.URLError) as failure:
            part.unlink(missing_ok=True)
            error = failure
        if attempt + 1 < attempts:
            time.sleep(min(BACKOFF_CAP_SECONDS, 2.0 * 2 ** attempt) * (0.5 + random.random()))
    raise RuntimeError(f"hydro source download failed after {attempts} attempts: {url}: {error}")


_ABSENT_LOCK = threading.Lock()


def _absent(directory):
    try:
        return {line.strip() for line in (Path(directory) / "ABSENT.txt").read_text().splitlines()}
    except OSError:
        return set()


def _mark_absent(path):
    with _ABSENT_LOCK:
        if path.name not in _absent(path.parent):
            with (path.parent / "ABSENT.txt").open("a") as listing:
                listing.write(path.name + "\n")


def rosters_table_path():
    return Path(__file__).resolve().parents[1] / "data" / "hydro" / "land-tile-rosters.v1.json"


_CELL = re.compile(r"([NS])(\d{2})(?:_00_)?([EW])(\d{3})")


def _cell(name):
    """(lat, lon) of a tile's south-west corner from its published name."""
    found = _CELL.search(name)
    if found is None:
        return None
    ns, lat, ew, lon = found.groups()
    return (int(lat) * (1 if ns == "N" else -1), int(lon) * (1 if ew == "E" else -1))


def land_cells(root, progress=print):
    """Per roster, the set of 1-degree land cells, from its published tile list.

    The list is kept in ``<root>/rosters`` (fetched once, reused by every later
    run). A roster that cannot be fetched is left out: those sources are asked
    tile by tile as before, sea answering 403/404.
    """
    table = json.loads(rosters_table_path().read_text())
    cells = {}
    for name, roster in table["rosters"].items():
        path = Path(root) / "rosters" / f"{name}.txt"
        try:
            _download(roster["url"], path, timeout=120, attempts=3)
            cells[name] = {cell for cell in map(_cell, path.read_text().split()) if cell is not None}
        except (OSError, RuntimeError) as error:
            progress(f"hydro sources: land roster {name} unavailable ({error}); asking its tiles one by one")
    return table["sources"], cells


def _over_land(url, sources, cells, lat=None, lon=None):
    """False only when the source's roster says the tile's cells are all sea."""
    for source in sources:
        if url.startswith(source["url_prefix"]) and source["roster"] in cells:
            if lat is None:
                corner = _cell(url.rsplit("/", 1)[-1])
                if corner is None:
                    return True
                lat, lon = corner
            step = int(source["tile_degrees"])
            land = cells[source["roster"]]
            return any((lat + i, lon + j) in land for i in range(step) for j in range(step))
    return True


def _geog_soilcomp(bbox=None):
    """The WPS_GEOG soilgrids/soilcomp directory, when this install has it
    (and, given ``bbox``, every tile of it the box plus 0.25 degree touches)."""
    try:
        from woof.geog_assets import default_geog_root
        path = Path(default_geog_root()) / "soilgrids" / "soilcomp"
        if not (path / "index").is_file():
            return None
        if bbox is None:
            return path
        index = dict(line.split("=", 1) for line in (path / "index").read_text().splitlines() if "=" in line)
        number = lambda key: float(index[key].strip().strip('"'))  # noqa: E731
        dx, dy, tx, ty = number("dx"), number("dy"), int(number("tile_x")), int(number("tile_y"))
        lon0, lat0 = number("known_lon"), number("known_lat")
        west, south, east, north = bbox
        if west > east:
            return None
        cols = [round((v - lon0) / dx) for v in (west - 0.25, east + 0.25)]
        rows = [round((v - lat0) / dy) for v in (south - 0.25, north + 0.25)]
        for tr in range(max(rows[0], 0) // ty, rows[1] // ty + 1):
            for tc in range(max(cols[0], 0) // tx, cols[1] // tx + 1):
                x0, y0 = tc * tx + 1, tr * ty + 1
                if not (path / f"{x0:05d}-{x0 + tx - 1:05d}.{y0:05d}-{y0 + ty - 1:05d}").is_file():
                    return None
        return path
    except Exception:  # noqa: BLE001 - no usable geography: SoilGrids WCS
        return None


def _extract(archive, destination):
    destination = Path(destination)
    stamp = destination / f".{Path(archive).name}.extracted"
    if stamp.is_file():
        return
    staging = destination.with_name(destination.name + f".partial.{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    if Path(archive).suffix.lower() == '.7z':
        try:
            import py7zr
        except ImportError as error:
            # LGPL-2.1+ decoder, an explicit extra (pyproject [hydro-conus]);
            # the door falls back to the worldwide fabric on this refusal.
            raise RuntimeError(f"{Path(archive).name} is a 7z archive; reading it needs the "
                               "LGPL-2.1+ py7zr decoder: pip install 'recast-woof[hydro-conus]'") from error
        bundle = py7zr.SevenZipFile(archive)
    else:
        bundle = zipfile.ZipFile(archive)
    with bundle:
        root = staging.resolve()
        names = bundle.getnames() if hasattr(bundle, 'getnames') else bundle.namelist()
        for member in names:
            target = (staging / member).resolve()
            if not target.is_relative_to(root):
                raise RuntimeError(f"archive member escapes its directory: {archive}: {member}")
        bundle.extractall(staging)
    destination.mkdir(parents=True, exist_ok=True)
    for item in staging.iterdir():
        target = destination / item.name
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        os.replace(item, target)
    shutil.rmtree(staging, ignore_errors=True)
    stamp.write_text(str(archive) + "\n")


def _archive_jobs(source_table, rows, regions=()):
    """Archive inputs (lakes, dams) of the builder rows: url + ``<dir>/extracted/...``."""
    jobs = {}
    for row in source_table["rows"]:
        if row["id"] not in rows:
            continue
        for name, spec in row["inputs"].items():
            for asset in spec.get('archives', ()):
                if asset['region'] in regions:
                    jobs[(asset['extracted'], asset['url'])] = dict(asset, input=name)
            url, local = spec.get("url"), spec.get("local")
            if not url or not local or "{" in url or "/extracted/" not in local:
                continue
            folder = local.split("/extracted/")[0]
            archive = url.rstrip("/").rsplit("/", 1)[-1]
            if not archive.endswith((".zip", ".7z")):
                # A download id (figshare) names its archive in the row's format text.
                named = [word for word in spec.get("format", "").split() if word.endswith(".zip")]
                archive = named[0] if named else f"{name}.zip"
            jobs[(folder, url)] = {"input": name, "url": url, "archive": f"{folder}/{archive}",
                                   "extracted": f"{folder}/extracted", "local": local}
    return list(jobs.values())


def _unpack(job):
    archive, destination = job
    start = time.perf_counter()
    _extract(archive, destination)
    return time.perf_counter() - start


def _unpack_archives(root, archives):
    """Independent archive processes use all cores, including 7z filters."""
    jobs = list({item['extracted']: (root / item['archive'], root / item['extracted'])
                 for item in archives}.values())
    if not jobs:
        return []
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        timings = list(pool.map(_unpack, jobs))
    for item in archives:
        if not (root / item['local']).exists():
            raise RuntimeError(f"archive {item['archive']} has no {item['local']}")
    return [{'archive': str(path), 'seconds': seconds} for (path, _), seconds in zip(jobs, timings)]


def _partition_archives(source_table, row_id, root, bbox, builder_exe):
    """Acquire the boundary roster first; Rust selects the touched partitions."""
    row = next(row for row in source_table['rows'] if row['id'] == row_id)
    if 'processing_units' not in row['inputs']:
        return [], []
    common = _archive_jobs(source_table, {row_id})
    boundary = [item for item in common if item['input'] == 'processing_units']
    for item in boundary:
        _download(item['url'], root / item['archive'])
    _unpack_archives(root, boundary)
    argv = [str(builder_exe), 'regions', '--sources', str(source_table_path()), '--row', row_id,
            '--data-root', str(root), '--bbox', ','.join(map(str, bbox))]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f"hydro processing-unit selection: {result.stderr[-1000:]}")
    regions = json.loads(result.stdout)
    if not regions:
        raise RuntimeError('no native processing unit intersects this footprint')
    return regions, _archive_jobs(source_table, {row_id}, regions)


def _builder_rows(source_table, requested):
    if requested == "auto":
        return {rule["row"] for rule in source_table.get("auto", ())}
    return {requested}


def _dem_tiles(exe, table, row, bbox, root):
    argv = [str(exe), "tiles", "--sources", str(table), "--row", row,
            "--bbox", ",".join(repr(float(v)) for v in bbox), "--data-root", str(root)]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f"rw_hydro_world tiles failed: {(result.stderr or result.stdout).strip()[-1000:]}")
    return json.loads(result.stdout)


def _soil_box(bbox, scale=1):
    west, south, east, north = bbox
    if west > east:
        raise RuntimeError("SoilGrids subset across the dateline is not supported by the WCS request; "
                           "split the parameter domain (owner hydro-world-params)")
    q = 0.25
    box = (math.floor(west / q) * q - q, math.floor(south / q) * q - q,
           math.ceil(east / q) * q + q, math.ceil(north / q) * q + q)
    box = (max(box[0], -180.0), max(box[1], -90.0), min(box[2], 180.0), min(box[3], 90.0))
    width, height = box[2] - box[0], box[3] - box[1]
    res = SOIL_BASE_DEG
    if (width / res) * (height / res) > SOIL_MAX_PIXELS:
        res = math.sqrt(width * height / SOIL_MAX_PIXELS)
        res = math.ceil(res / SOIL_BASE_DEG) * SOIL_BASE_DEG
    res *= scale
    nx, ny = int(round(width / res)), int(round(height / res))
    tag = "_".join(f"{v:+.2f}".replace("+", "p").replace("-", "m").replace(".", "d") for v in box)
    return box, nx, ny, f"bbox_{tag}_{nx}x{ny}"


def acquire(*, footprint, data_root, builder_exe, builder_row, recipe, progress=print):
    """Fetch what the footprint needs and return the recipe token bindings.

    ``recipe`` is the parameter recipe template (with ``{data_root}``,
    ``{params_table}`` and ``{soil_prefix}`` tokens); its ``dem``, ``soil``
    and ``land_cover`` source ids select rows of the params table.
    """
    from woof.remote_artifact_cache import Lease

    root = Path(data_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    bbox = [footprint[k] for k in ("west", "south", "east", "north")]
    source_table = json.loads(source_table_path().read_text())
    params = json.loads(params_table_path().read_text())
    partition_started = time.perf_counter()
    partition_archives, regions = [], []
    if builder_row != 'auto':
        with Lease(root / '.partition.lock', timeout=float('inf')):
            regions, partition_archives = _partition_archives(source_table, builder_row, root, bbox, builder_exe)
    partition_seconds = time.perf_counter() - partition_started
    sources, cells = land_cells(root, progress)
    jobs = []  # (kind, url, path, absent_ok)
    sea = []   # tiles the land rosters place over open sea: never requested
    for tile in _dem_tiles(builder_exe, source_table_path(), builder_row, bbox, root):
        if not tile["exists"]:
            if _over_land(tile["url"], sources, cells):
                jobs.append(("builder-dem", tile["url"], root / tile["local"], True))
            else:
                sea.append(("builder-dem", root / tile["local"]))
    archives = partition_archives or _archive_jobs(source_table, _builder_rows(source_table, builder_row))
    for item in archives:
        jobs.append(("archive", item["url"], root / item["archive"], False))
    margin = 0.05
    south, north = bbox[1] - margin, bbox[3] + margin
    west, east = bbox[0] - margin, bbox[2] + margin
    dem = params["dem_sources"][recipe["dem"]["source"]]
    dem_dir = Path(recipe["dem"]["dir"].replace("{data_root}", str(root)))
    for lat, lon in _tiles(south, north, west, east, 1):
        name = _fill_tile(dem["tile_pattern"], lat, lon)
        url = _fill_tile(dem["url_pattern"], lat, lon)
        if _over_land(url, sources, cells, lat, lon):
            jobs.append(("params-dem", url, dem_dir / name, True))
        else:
            sea.append(("params-dem", dem_dir / name))
    cover = params["land_cover_sources"][recipe["land_cover"]["source"]]
    cover_dir = Path(recipe["land_cover"]["dir"].replace("{data_root}", str(root)))
    step = int(cover.get("tile_degrees", 1))
    for lat, lon in _tiles(south, north, west, east, step):
        name = _fill_tile(cover["tile_pattern"], lat, lon)
        url = _fill_tile(cover["url_pattern"], lat, lon)
        if _over_land(url, sources, cells, lat, lon):
            jobs.append(("land-cover", url, cover_dir / name, True))
        else:
            sea.append(("land-cover", cover_dir / name))
    soil = params["soil_sources"][recipe["soil"]["source"]]
    soil_dir = Path(recipe["soil"]["dir"].replace("{data_root}", str(root)))

    def soil_plan(scale):
        box, nx, ny, prefix = _soil_box(bbox, scale=scale)
        planned = []
        for prop in soil["props"]:
            for depth in soil["depth_labels"]:
                url = (soil["url_pattern"].replace("{prop}", prop).replace("{depth}", depth)
                       .replace("long(W,E)", f"long({box[0]:.4f},{box[2]:.4f})")
                       .replace("lat(S,N)", f"lat({box[1]:.4f},{box[3]:.4f})")
                       .replace("long(NX)", f"long({nx})").replace("lat(NY)", f"lat({ny})"))
                name = (soil["file_pattern"].replace("{prefix}", prefix).replace("{prop}", prop)
                        .replace("{depth}", depth))
                planned.append(("soil", url, soil_dir / name, False))
        return (box, nx, ny, prefix), planned

    soil_scale = 1
    (box, nx, ny, prefix), soil_jobs = soil_plan(soil_scale)
    # Our own geography mirror first: WPS_GEOG soilgrids/soilcomp (SoilGrids-
    # derived sand and clay, 30 arc-seconds) when it holds every tile the box
    # needs; the ISRIC WCS only where it does not. Breakage it prevents: an
    # ISRIC outage (504 on every request from ~16:50Z 2026-10-21) took hydro
    # off every default run.
    primary = _geog_soilcomp(bbox)
    if primary is not None:
        soil_jobs = []
    # The builder and the parameter recipe read the same GLO-90 tiles from the
    # same directory: each file is asked once (two jobs for one path used to
    # download it twice, into one .part file from two threads).
    unique = {}
    for job in jobs:
        unique.setdefault(job[2], job)
    jobs = list(unique.values())
    sea = list({path: (kind, path) for kind, path in sea}.values())
    # One acquirer per cache root; concurrent launchers wait and then reuse.
    lock = Lease(root / ".acquire.lock")
    if lock.file is None:
        progress(f"hydro sources: waiting for another acquisition in {root}")
        lock = Lease(root / ".acquire.lock", timeout=float("inf"))
    with lock:
        counts = {}
        fetched_bytes = 0
        for kind, path in sea:
            if not path.is_file():
                path.parent.mkdir(parents=True, exist_ok=True)
                _mark_absent(path)
            counts.setdefault(kind, {}).setdefault("sea", 0)
            counts[kind]["sea"] += 1

        # A host that keeps failing (every retry of a request exhausted, with
        # no success from it yet) is down: its other requests are not sent.
        # Breakage it prevents: during the ISRIC outage of 2026-10-21 every
        # SoilGrids request answered 504 in 15 s, and 36 requests with
        # retries would have held the preparation for most of an hour.
        down, alive, hosts = set(), set(), threading.Lock()

        def run(job):
            kind, url, path, absent_ok = job
            host = urllib.parse.urlsplit(url).netloc
            with hosts:
                if host in down and host not in alive:
                    return kind, "failed", 0, (job, f"{host} is not answering (an earlier request to it failed every retry): {url}")
            try:
                status, size = _download(url, path, absent_ok=absent_ok)
            except RuntimeError as error:
                with hosts:
                    down.add(host)
                return kind, "failed", 0, (job, str(error))
            with hosts:
                alive.add(host)
            return kind, status, size, None

        def fetch_all(tiles, soils):
            with ThreadPoolExecutor(max_workers=WORKERS) as pool, \
                    ThreadPoolExecutor(max_workers=SOIL_WORKERS) as soil_pool:
                soil_futures = [soil_pool.submit(run, job) for job in soils]
                done = list(pool.map(run, tiles))
                return done + [future.result() for future in soil_futures]

        progress(f"hydro sources: {len(jobs) + len(soil_jobs)} inputs for "
                 f"[{', '.join(f'{v:.2f}' for v in bbox)}] under {root} "
                 f"({len(sea)} sea tiles never requested)")
        results = fetch_all(jobs, soil_jobs)
        retried = []
        # One more round for what failed, after everything else is in: a
        # gateway timeout on one tile does not abandon the other thousand.
        failed = [r[3][0] for r in results if r[3] is not None
                  and not (urllib.parse.urlsplit(r[3][0][1]).netloc in down
                           and urllib.parse.urlsplit(r[3][0][1]).netloc not in alive)]
        if failed:
            progress(f"hydro sources: {len(failed)} input(s) failed; retrying them once more")
            retried = fetch_all([j for j in failed if j[0] != "soil"], [j for j in failed if j[0] == "soil"])
            results = [r for r in results if r[3] is None] + retried
        # SoilGrids that still times out is asked for the same box on a
        # coarser grid (half the pixels each time, at most twice): the WCS
        # serves a smaller subset where a large one keeps timing out.
        degraded = []
        soil_host = urllib.parse.urlsplit(soil_jobs[0][1]).netloc if soil_jobs else ""
        while (any(r[3] is not None and r[0] == "soil" for r in results) and soil_scale < 4
               and soil_host in alive):
            soil_scale *= 2
            (box, nx, ny, prefix), soil_jobs = soil_plan(soil_scale)
            degraded.append({"scale": soil_scale, "nx": nx, "ny": ny})
            progress(f"hydro sources: SoilGrids still failing; asking {nx} x {ny} (1/{soil_scale} of the pixels)")
            results = [r for r in results if r[0] != "soil"] + fetch_all([], soil_jobs)
        # SoilGrids out of reach: the parameters read the SoilGrids-derived sand
        # and clay on the WPS_GEOG mirror every install has (rw_hydro_params
        # geog_soilcomp), so hydro still attaches. Breakage it prevents: the
        # ISRIC outage of 2026-10-21 took hydro off every default run.
        geog_soilcomp = str(primary) if primary is not None else ""
        soil_failures = [r for r in results if r[3] is not None and r[0] == "soil"]
        if soil_failures:
            fallback = _geog_soilcomp()
            if fallback is not None:
                geog_soilcomp = str(fallback)
                progress(f"hydro sources: FALLBACK soil -- {len(soil_failures)} SoilGrids request(s) failed "
                         f"({soil_failures[0][3][1][-160:]}); the parameters read {fallback}")
                results = [r for r in results if not (r[3] is not None and r[0] == "soil")]
        failures = [r[3][1] for r in results if r[3] is not None]
        if failures:
            unreachable = sorted(down - alive)
            raise RuntimeError(f"hydro sources: {len(failures)} input(s) failed after retries"
                               + (f" ({', '.join(unreachable)} not answering)" if unreachable else "")
                               + "; the inputs already fetched stay in " + str(root) + " for the next run: "
                               + "; ".join(failures[:2]))
        for kind, status, size, _ in results:
            counts.setdefault(kind, {}).setdefault(status, 0)
            counts[kind][status] += 1
            fetched_bytes += size
        unpack_start = time.perf_counter()
        unpack = _unpack_archives(root, archives)
        unpack_seconds = time.perf_counter() - unpack_start
    receipt = {"schema": "gpuwm-hydro-acquire.v1", "data_root": str(root), "bbox": bbox,
               "counts": counts, "fetched_bytes": fetched_bytes,
               "processing_units": regions, "partition_seconds": partition_seconds,
               "unpack_seconds": unpack_seconds, "unpack": unpack,
               "unpack_parallelism": f"{os.cpu_count() or 1} archive processes (all cores)",
               "soil": ({"source": "WPS_GEOG soilgrids/soilcomp (our geography mirror)", "dir": geog_soilcomp,
                         **({"reason": "SoilGrids WCS out of reach"} if primary is None else {})}
                        if geog_soilcomp else
                        {"source": "SoilGrids WCS", "box": box, "nx": nx, "ny": ny, "prefix": prefix,
                         **({"degraded_after_timeouts": degraded} if degraded else {})}),
               "land_rosters": sorted(cells), "retried_inputs": len(retried),
               "seconds": time.perf_counter() - started,
               "parallelism": f"{WORKERS} transfer threads + {SOIL_WORKERS} WCS requests (network bound)"}
    progress(f"hydro sources: ready in {receipt['seconds']:.1f} s, "
             f"{fetched_bytes / 1e9:.2f} GB fetched; {json.dumps(counts, sort_keys=True)}")
    return {"data_root": str(root), "params_table": str(params_table_path()),
            "soil_prefix": prefix, "geog_soilcomp": geog_soilcomp}, receipt


def bind_recipe(recipe, tokens):
    """Replace the template tokens in every string of the recipe."""
    def visit(value):
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, str):
            for key, item in tokens.items():
                value = value.replace("{" + key + "}", str(item))
        return value
    return visit(recipe)
