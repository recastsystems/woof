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

def prepare_grid(*, grid, start_time, end_time, output_root, options=None):
    """Prepare any supported grid, acquiring public inputs when no recipe is supplied."""
    options = dict(options or {})
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
