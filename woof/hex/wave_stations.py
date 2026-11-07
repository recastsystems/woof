"""Native open-water station selection from prepared model directions and mesh mask."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def audit_binary():
    """Resolve the required station helper in an install or source checkout."""
    from .engines import bundled_program, executable_name
    explicit = os.environ.get("WOOF_HEX_RW_WAVE_AUDIT")
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError("named hex wave station helper is absent: " + explicit)
        return path
    packaged = bundled_program("wave_audit")
    if packaged:
        return Path(packaged)
    products = os.environ.get("WOOF_HEX_RW_WAVE_PRODUCTS")
    if products:
        sibling = Path(products).with_name(executable_name("wave_audit"))
        if sibling.is_file():
            return sibling
    crate = Path(__file__).resolve().parents[2] / "rust" / "wave-products"
    built = crate / "target" / "release" / executable_name("wave_audit")
    if built.is_file():
        return built
    found = shutil.which("wave_audit")
    if found:
        return Path(found)
    raise FileNotFoundError(
        "hex wave station helper wave_audit is absent from woof/hex/libexec; "
        f"source builds use cargo build --release --manifest-path {crate / 'Cargo.toml'}")


def exposed_stations(path, manifest, output_root):
    """Keep buoy coordinates and record the selected cell and its actual offset.

    The prepared reference is a model forecast, never observations. Testing its
    incoming directions prevents a nearest coastal cell from representing an
    offshore buoy behind the coarse mesh's unresolved coast. If no cell is fully
    exposed within the declared radius, retain the best available exposure and
    label the limitation in the station record.
    """
    import numpy as np
    from woof.netcdf_bridge import open_dataset
    path = Path(path).resolve()
    stations = manifest["stations"]
    if not stations:
        return stations, None
    root = Path(output_root).resolve() / "waves" / "station-mapping"
    root.mkdir(parents=True, exist_ok=True)
    reference = path / "reference" / "points.nc"
    with open_dataset(reference) as dataset:
        ids = [bytes(row).decode("ascii").strip("\x00 ") for row in dataset.variables["station_id"][:]]
        if ids != [s["id"] for s in stations]:
            raise ValueError("prepared reference station identifiers/order differs")
        direction = dataset.variables["dp"]
        if direction.shape[0] != len(stations):
            raise ValueError("prepared reference station order/extent differs")
        direction_times = direction.shape[1]
        # Rust decodes the NetCDF; Python only packs the native audit wire input.
        (root / "directions.bin").write_bytes(np.ascontiguousarray(direction[:], dtype="<f4").tobytes())
    def array(name):
        spec = manifest["arrays"][name]
        return dict(path=str(path/spec["path"]), dtype=spec["dtype"].lstrip("<"), bytes=spec["bytes"])
    request = dict(operation="stations", latitude=array("lat"), longitude=array("lon"),
        wet=array("wet"), depth=array("depth"), area=array("area"), stations=stations,
        max_distance_km=manifest.get("request", {}).get("station_max_distance_km", 75.),
        directions=str(root/"directions.bin"), direction_times=direction_times,
        threads=os.cpu_count() or 1, output=str(root/"result.json"))
    recipe = root / "request.json"
    recipe.write_text(json.dumps(request, indent=2)+"\n")
    executable = audit_binary()
    subprocess.run([executable, str(recipe)], check=True, timeout=600)
    raw = (root/"result.json").read_bytes()
    result = json.loads(raw)
    mapped = []
    for station, row in zip(stations, result["stations"], strict=True):
        selected = row["selected"]
        if selected is None:
            raise ValueError(f"no wet wave station cell within radius: {station['id']}")
        mapped.append(dict(station, nearest_cell=station["cell"], cell=selected["cell"],
            distance_km=selected["distance_km"], model_lat=selected["lat"], model_lon=selected["lon"],
            model_depth_m=selected["depth_m"], fully_exposed=row["fully_exposed"],
            exposed_fraction=selected["exposed_fraction"], ray_extent_km=row["ray_extent_km"],
            selection_rule=row["rule"]))
    return mapped, hashlib.sha256(raw).hexdigest()
