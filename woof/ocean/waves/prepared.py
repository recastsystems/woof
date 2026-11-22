"""Common native preparation API for atmospheric and Voronoi wave callers."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from .data import invoke

# All preparation controls travel to Rust. Solver options remain with the caller.
_PREPARATION_OPTIONS = (
    "station_max_distance_km", "coast_fill_cells", "source_catalog", "input_cache",
    "shared_inputs", "stations", "coarsening", "field_table", "bathymetry",
    "timeout_seconds",
)

def _input_cache_root():
    """Host cache for wave inputs every run reads alike: ``$WOOF_WAVES_INPUT_CACHE``, else
    ``$WOOF_WAVES_CACHE/shared-inputs``, else ``~/.woof/cache/waves/shared-inputs``."""
    explicit = os.environ.get("WOOF_WAVES_INPUT_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    root = os.environ.get("WOOF_WAVES_CACHE")
    return (Path(root).expanduser() if root else Path.home() / ".woof" / "cache" / "waves") / "shared-inputs"


def _sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def shared_bathymetry(options):
    """One checksummed bathymetry copy per host, staged once and reused by every later preparation.

    THE BREAKAGE THIS PREVENTS: each preparation downloaded its own 491 MB ETOPO 2022 copy into its own output
    tree; two runs started seconds apart were rate-limited by NOAA NCEI (HTTP 429) and the second stopped
    ("bathymetry acquisition incomplete", woof hex, 2026-11-06). The first run stages it through the Rust
    ``stage`` door (partial file, sha256 receipt, rename into place) under a lock on the cache entry; a run
    arriving meanwhile waits, then reuses the file once its sha256 matches the receipt (a damaged copy is
    fetched again). Options that already name a bathymetry, a shared one or a catalog of their own are kept.
    """
    shared = dict(options.get("shared_inputs") or {})
    if "bathymetry" in shared or options.get("bathymetry") or options.get("source_catalog"):
        return options
    from importlib.resources import files
    from woof.ocean.automatic import _lock
    row = json.loads(files("woof.ocean.waves.data").joinpath("public_sources.v1.json").read_text())["bathymetry"]
    key = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:20]
    folder = _input_cache_root() / "bathymetry" / key
    target = folder / row["path"]
    receipt = target.with_name(target.name + ".receipt.json")

    def verified():
        acquired = folder / "ACQUIRED.json"
        if not (target.is_file() and acquired.is_file() and receipt.is_file()):
            return False
        expected = row.get("expected_bytes")
        return (json.loads(acquired.read_text()).get("status") == "complete"
                and (expected is None or target.stat().st_size == int(expected))
                and json.loads(receipt.read_text()).get("sha256") == _sha256(target))

    folder.parent.mkdir(parents=True, exist_ok=True)
    with _lock(folder.parent / f"{key}.lock"):
        if not verified():
            for stale in (target, receipt, folder / "ACQUIRED.json"):
                stale.unlink(missing_ok=True)
            folder.mkdir(parents=True, exist_ok=True)
            request = folder / "stage-request.json"
            request.write_text(json.dumps({"operation": "stage", "output": str(folder), "start": "2000-01-01T00:00:00Z",
                                           "end": "2000-01-01T00:00:00Z", "stations": [], "sources": [row]},
                                          indent=2) + "\n")
            invoke(request, operation="stage")
            if not verified():
                raise RuntimeError(f"wave bathymetry acquisition into {folder} did not complete; see its ACQUIRED.json")
    shared["bathymetry"] = str(target)
    return dict(options, shared_inputs=shared)


def prepare_grid(*, grid, start_time, end_time, output_root, options=None):
    """Prepare any supported grid, acquiring public inputs when no recipe is supplied."""
    options = dict(options or {})
    if not (options.get("source_prepared") or os.environ.get("WOOF_WAVES_SOURCE_PREPARED")):
        options = shared_bathymetry(options)
    grid = dict(grid)
    if "file" in grid:
        source_grid = Path(grid["file"]).resolve()
        grid["file"] = str(source_grid)
        with source_grid.open("rb") as stream:
            grid["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    def stamp(value):
        from datetime import datetime, timezone
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()
    source = options.get("source_prepared") or os.environ.get("WOOF_WAVES_SOURCE_PREPARED")
    request = {"operation": "prepare-from" if source else "prepare", "grid": grid,
               "start": stamp(start_time), "end": stamp(end_time),
               "threads": options.get("prepare_threads", os.cpu_count() or 1)}
    if source:
        request["source_prepared"] = str(Path(source).resolve())
    request.update({key: options[key] for key in _PREPARATION_OPTIONS if key in options})
    fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    out = Path(output_root).resolve() / "waves-prepared" / fingerprint[:20]
    request["output"] = str(out)
    recipe = out / "request.json"
    if (out / "manifest.json").exists():
        if not recipe.exists() or json.loads(recipe.read_text()) != request:
            raise ValueError("existing wave preparation request differs")
        manifest = json.loads((out / "manifest.json").read_text())
        if manifest.get("status") != "complete":
            raise ValueError("existing wave preparation is incomplete")
        return out
    out.mkdir(parents=True, exist_ok=True)
    recipe.write_text(json.dumps(request, indent=2) + "\n")
    return invoke(recipe, operation=request["operation"])

def prepare_mesh(*, mesh, grid_path, start_time, end_time, output_root, options=None):
    """Prepare in the static mesh's original cell order through the shared Rust door."""
    options = dict(options or {})
    grid = {"kind": "points", "file": str(grid_path),
            "latitude": options.get("latitude_variable", "latCell"),
            "longitude": options.get("longitude_variable", "lonCell"),
            "coordinate_units": options.get("coordinate_units", "radian"),
            "area": options.get("area_variable", "areaCell"),
            "boundary_connectivity": options.get("boundary_connectivity", "cellsOnEdge")}
    if options.get("boundary_index_variable"):
        grid["boundary_index_variable"] = options["boundary_index_variable"]
    if "boundary_cells" in options:
        grid["boundary_cells"] = options["boundary_cells"]
    return prepare_grid(grid=grid, start_time=start_time, end_time=end_time,
                        output_root=output_root, options=options)
