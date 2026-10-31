"""One Rust request door shared by structured and Voronoi wave callers."""
from __future__ import annotations
import ctypes
import hashlib
import json
import math
import mmap
from pathlib import Path
import subprocess

def native_binary():
    from woof import bridges
    found = bridges.find_artifact("GPUWM_RW_WAVES_DATA", bridges.executable_name("rw_waves_data"), crate_relative="tools/waves_data")
    if found is None:
        raise FileNotFoundError("rw_waves_data is required; build tools/waves_data or set GPUWM_RW_WAVES_DATA to the built bridge")
    return found

def invoke(request, *, operation):
    path = Path(request).resolve()
    document = json.loads(path.read_text())
    if document.get("operation") != operation:
        raise ValueError(f"request operation must be {operation}")
    subprocess.run([str(native_binary()), str(path)], check=True, timeout=document.get("timeout_seconds", 7200))
    return Path(document["output"])

def prepare(request):
    return invoke(request, operation="prepare")

def load_prepared(path, *, forcing=None):
    """Validate byte receipts and upload without Python numerical transforms.

    Geometry and partition arrays are in full-cell order. Boundary indices
    name full cells, not the core backend's compact wet list.
    """
    import cupy as cp
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema") != "gpuwm.waves.prepared/v1":
        raise ValueError("wrong wave prepared schema")
    sources = {name: (path, row) for name, row in manifest["arrays"].items()}
    if forcing is not None:
        forcing = Path(forcing)
        alternate = json.loads((forcing / "manifest.json").read_text())
        if alternate.get("schema") != "gpuwm.waves.forcing/v1" or alternate.get("status") != "complete":
            raise ValueError("alternate wave forcing is incomplete")
        with (path / "manifest.json").open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if alternate["prepared_manifest_sha256"] != digest or alternate["forcing_times"] != manifest["forcing_times"]:
            raise ValueError("alternate forcing belongs to a different grid or time window")
        for name in ("u10_east", "v10_north"):
            row = alternate["arrays"][name]
            if row["shape"] != manifest["arrays"][name]["shape"]:
                raise ValueError("alternate forcing shape differs")
            sources[name] = (forcing, row)
        manifest = {**manifest, "alternate_forcing": alternate}
    arrays = {}
    for name, (root, row) in sources.items():
        source = (root / row["path"]).resolve()
        if not source.is_relative_to(root.resolve()):
            raise ValueError("prepared array path escapes its folder")
        if row["dtype"] not in ("<f4", "<i4"):
            raise ValueError("wave arrays must be little-endian float32 or int32")
        with source.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != row["sha256"]:
                raise ValueError(f"wave input hash differs: {name}")
        size = 4 * math.prod(row["shape"])
        if source.stat().st_size != size:
            raise ValueError(f"wave input byte count differs: {name}")
        array = cp.empty(tuple(row["shape"]), dtype=row["dtype"])
        if size:
            with source.open("rb") as stream, mmap.mmap(stream.fileno(), size, access=mmap.ACCESS_COPY) as data:
                pointer = ctypes.addressof(ctypes.c_char.from_buffer(data))
                cp.cuda.runtime.memcpy(array.data.ptr, pointer, size, cp.cuda.runtime.memcpyHostToDevice)
        arrays[name] = array
    return manifest, arrays

def register_cli(commands):
    parser = commands.add_parser("waves", help="prepare, verify and render spectral wave data")
    doors = parser.add_subparsers(dest="wave_data_command", required=True)
    for name in ("stage", "catalog", "prepare", "forcing", "audit", "verify", "products", "observations"):
        door = doors.add_parser(name)
        door.add_argument("request", type=Path)
        door.set_defaults(func=main, wave_data_operation=name)

def main(args):
    if args.wave_data_operation == "verify":
        from ..verify import run
        run(args.request)
    elif args.wave_data_operation == "products":
        from ..products import run
        run(args.request)
    else:
        invoke(args.request, operation=args.wave_data_operation)
    return 0
