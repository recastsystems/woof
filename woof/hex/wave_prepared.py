"""Transfer the shared Rust prepared contract to CUDA, without field decoding."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import threading

_PREPARE_LOCK = threading.Lock()   # where there is no fcntl, the rank threads of one process still take turns


@contextmanager
def _locked(path, fallback):
    """An exclusive fcntl lock on ``path`` (a process-level lock where there is no fcntl)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:
        with fallback:
            yield
        return
    with open(path, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


_BATHYMETRY_LOCK = threading.Lock()


def _input_cache_root():
    """Where wave inputs every run reads alike are kept: ``$WOOF_HEX_WAVES_INPUT_CACHE``, else
    ``$WOOF_HEX_CACHE_ROOT/waves-inputs``, else ``~/.cache/gpuwm-hex/waves-inputs``."""
    explicit = os.environ.get("WOOF_HEX_WAVES_INPUT_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    root = os.environ.get("WOOF_HEX_CACHE_ROOT")
    return (Path(root).expanduser() if root else Path.home() / ".cache" / "woof hex") / "waves-inputs"


def shared_bathymetry(options):
    """One bathymetry download per cache entry on this host, reused by every later run.

    THE BREAKAGE THIS PREVENTS: each run's preparation downloaded its own 491 MB ETOPO 2022 copy into its
    own output tree, in parallel 8 MB ranges. Two runs started on one box within seconds of each other: the
    NOAA NCEI server answered the second with HTTP 429 and that run stopped ("bathymetry acquisition
    incomplete", 2026-11-06). Now the first run acquires it once, under an fcntl lock on the cache entry
    (the shared Rust ``stage`` door: partial file, sha256 receipt, rename into place). A run that arrives
    meanwhile waits on the lock, then reuses the finished file once its sha256 matches that receipt (a damaged
    copy is fetched again), and the preparation reads it through
    ``shared_inputs``. Options that already name a bathymetry, or a source catalog of their own, are left as given.
    """
    shared = dict(options.get("shared_inputs") or {})
    if "bathymetry" in shared or options.get("source_catalog"):
        return options
    from importlib.resources import files
    catalog = json.loads(files("woof.ocean.waves.data").joinpath("public_sources.v1.json").read_text())
    row = catalog["bathymetry"]
    key = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:20]
    folder = _input_cache_root() / "bathymetry" / key
    target = folder / row["path"]
    receipt = target.with_name(target.name + ".receipt.json")

    def complete():
        # Reuse only a copy whose sha256 still matches the stage door's receipt; a damaged one is fetched again.
        acquired = folder / "ACQUIRED.json"
        if not (target.is_file() and acquired.is_file() and receipt.is_file()):
            return False
        expected = row.get("expected_bytes")
        if not (json.loads(acquired.read_text()).get("status") == "complete"
                and (expected is None or target.stat().st_size == int(expected))):
            return False
        with open(target, "rb") as stream:
            return json.loads(receipt.read_text()).get("sha256") == hashlib.file_digest(stream, "sha256").hexdigest()

    folder.parent.mkdir(parents=True, exist_ok=True)
    with _locked(folder.parent / f"{key}.lock", _BATHYMETRY_LOCK):
        if not complete():
            for stale in (target, receipt, folder / "ACQUIRED.json"):
                stale.unlink(missing_ok=True)
            from woof.ocean.waves.data import invoke
            folder.mkdir(parents=True, exist_ok=True)
            request = folder / "stage-request.json"
            request.write_text(json.dumps({"operation": "stage", "output": str(folder), "start": "2000-01-01T00:00:00Z",
                                           "end": "2000-01-01T00:00:00Z", "stations": [], "sources": [row]}, indent=2) + "\n")
            invoke(request, operation="stage")
            if not complete():
                raise RuntimeError(f"wave bathymetry acquisition into {folder} did not complete; see its ACQUIRED.json")
    shared["bathymetry"] = str(target)
    return dict(options, shared_inputs=shared)


@contextmanager
def _one_preparer(output_root):
    """One wave preparation at a time per output root, across rank threads and rank processes.

    THE BREAKAGE THIS PREVENTS: every same-host rank calls ``ensure`` at the same moment with the same output root.
    The shared preparation writes one recipe directory and downloads the bathymetry into it, so a second rank
    found the file not yet there and stopped the run ("bathymetry acquisition incomplete", 2 rank processes on
    x1.40962, waves on, 2026-11-06). Holding the lock, the first rank prepares. Each later rank then finds the
    finished preparation (its manifest), which ``prepare_grid`` reuses without preparing it again.
    """
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:
        with _PREPARE_LOCK:
            yield
        return
    with open(root / ".waves-prepare.lock", "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@dataclass
class Prepared:
    path: Path
    manifest: dict
    arrays: dict
    sha256: str
    content_sha256: str = ""


#: Manifest keys that describe the preparation run itself, not what it prepared: where it wrote (request and stage
#: output paths, the field table's path), how many threads, and its wall times.
_RUN_ONLY_KEYS = ("request", "stages", "parallelism")


def content_sha256(manifest):
    """The digest of what a preparation holds: every array (each with its own sha256), grid, times and stations.

    THE BREAKAGE THIS PREVENTS (hex 0.4.14): a wave checkpoint named its preparation by the sha256 of the manifest's
    raw bytes, which carry the run's own output directory and wall times.  A resume always prepares under its own
    output root, so the same recipe (same arrays, word for word) gave other bytes, and every wave resume into a new
    directory refused with "hex wave restart prepared inputs or clock origin differs".
    """
    kept = {key: value for key, value in manifest.items() if key not in _RUN_ONLY_KEYS}
    return hashlib.sha256(json.dumps(kept, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def coordinate_difference(mesh_radians, prepared_degrees, *, longitude=False, xp):
    """Compare two encodings of one coordinate within their rounding bounds.

    Hex stores mesh angles as float32 radians, while prepared/v1 exchanges
    float32 degrees. Compare in float64 and allow half an ulp from each
    stored encoding, instead of imposing a tolerance smaller than one ulp.
    This does not change either coordinate or regrid a field.

    The mesh half ulp is a float32 one even when the mesh carries float64
    angles (the WOOF route reads them from the float64 grid file): the
    preparation reads the float32 native static, so its points went through
    a float32 radian encoding either way, in the static's own range. THE BREAKAGE THIS PREVENTS: every
    ArWen-route run with --waves on over the native global x1.40962 mesh
    refused at service start (11,202 latitudes 7.0e-6 deg apart against a
    3.8e-6 deg allowance, 2026-11-06), although the static is the float32
    rounding of the grid word for word.
    """
    mesh = xp.rad2deg(mesh_radians.astype(xp.float64))
    prepared = prepared_degrees.astype(xp.float64)
    delta = mesh - prepared
    if longitude:
        delta = (delta + 180.) % 360. - 180.
    def ulp(value):
        away = xp.where(value < 0, -xp.inf, xp.inf).astype(value.dtype)
        return xp.abs(xp.nextafter(value, away) - value).astype(xp.float64)
    # A float64 mesh angle gets the rounding of the float32 static radian the preparation read, in that
    # file's own range (longitudes in [0, 2 pi)); a float32 mesh angle is that encoding already.
    encoded = (mesh_radians if mesh_radians.dtype == xp.float32 else
               xp.deg2rad(prepared_degrees.astype(xp.float64)).astype(xp.float32))
    tolerance = (ulp(encoded) * (.5 * 180. / 3.141592653589793)
                 + .5 * ulp(prepared_degrees) + 1.e-10)
    return xp.abs(delta), tolerance


def load(path):
    import cupy as cp
    from woof.ocean.waves.prepared_run import NativeIO
    path = Path(path).resolve()
    raw = (path / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema") != "gpuwm.waves.prepared/v1" or manifest.get("status") != "complete":
        raise ValueError("hex waves require completed shared spectral preparation")
    required = {"lat", "lon", "depth", "wet", "area", "initial_partitions",
                "boundary_cells", "boundary_partitions", "u10_east", "v10_north"}
    if required - manifest.get("arrays", {}).keys():
        raise ValueError("spectral prepared arrays are incomplete")

    io = NativeIO()
    # Shared Rust byte reader, with independent verification/reads parallel.
    # Rust owns the decode, units, regridding, mask and partition preparation.
    with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 1, 24)) as pool:
        names = sorted(required)
        hosts = list(pool.map(lambda name:io.load(path,manifest["arrays"][name]), names))
    # Coupled stepping gets subsequent winds from the atmosphere. Only the
    # first prepared wind plane seeds initial partition diagnostics.
    arrays = {name:cp.asarray(host[0] if name in ("u10_east","v10_north") else host)
              for name,host in zip(names,hosts)}
    return Prepared(path, manifest, arrays, hashlib.sha256(raw).hexdigest(), content_sha256(manifest))


def ensure(*, mesh, grid_path, start_time, end_time, output_root, options):
    path = options.get("prepared")
    if path is None:
        try:
            from woof.ocean.waves.prepared import prepare_mesh
        except ImportError as exc:
            raise RuntimeError("hex wave automatic preparation requires the shared waves-data prepare_mesh entry point") from exc
        # That one shared entry point owns cache reuse, bathymetry, GFS-Wave,
        # stations and polygon-aware point-list preparation on Rust workers.
        preparation = options.get("preparation", {})
        if not isinstance(preparation, dict):
            raise ValueError("wave preparation options must be a JSON object")
        with _one_preparer(output_root):
            path = prepare_mesh(mesh=mesh, grid_path=grid_path, start_time=start_time,
                end_time=end_time, output_root=output_root,
                options=shared_bathymetry(dict(options, **preparation)))
    from .wave_stations import exposed_stations
    manifest = json.loads((Path(path) / "manifest.json").read_text())
    stations, mapping_sha = exposed_stations(path, manifest, output_root)
    prepared = load(path)
    prepared.manifest = dict(prepared.manifest, stations=stations, station_mapping_sha256=mapping_sha)
    return prepared
