"""NextGen hydrofabric startup orchestration; SQLite, geometry and payload I/O are Rust."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from woof import bridges

CRATE = "tools/nextgen_hydrofabric"
BINARY_ENV = "WOOF_RW_HYDROFABRIC"
LIBRARY_ENV = "WOOF_HYDROFABRIC_LIBRARY"
SCHEMA = "arwen.nextgen.hydrofabric.v1"


def _artifact(env: str, filename: str) -> Path:
    candidates = bridges.artifact_candidates(env, filename, crate_relative=CRATE)
    override = os.environ.get(env)
    for path in candidates:
        if path.is_file():
            return bridges.accept_resolved(path.resolve(), executable=env == BINARY_ENV)
        if override and path == Path(override):
            raise FileNotFoundError(f"{env} names a missing file: {path}")
    if env == LIBRARY_ENV:
        # On demand since 2.9.11 (PyPI per-file limit): pinned in this
        # release's bundle, staged by the first hydro run that loads it.
        if not override and bridges.fetch_on_demand("nextgen_hydrofabric"):
            for path in bridges.artifact_candidates(env, filename, crate_relative=CRATE):
                if path.is_file():
                    return bridges.accept_resolved(path.resolve(), executable=False)
        raise FileNotFoundError(bridges.artifact_remedy(
            env_var=env, filename=filename, subject="NextGen Rust payload transport",
            crate_relative=CRATE, artifact="nextgen_hydrofabric"))
    raise FileNotFoundError(
        f"Missing Rust hydrofabric artifact {filename}. "
        f"Build {CRATE} with cargo build --release --offline, or set {env}. "
        "The CLI currently requires system SQLite and PROJ development libraries."
    )


def prepare(config: str | Path) -> dict:
    exe = _artifact(BINARY_ENV, "rw_hydrofabric.exe" if os.name == "nt" else "rw_hydrofabric")
    result = subprocess.run([str(exe), "prepare", str(config)], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def prepare_cell_overlap(config: str | Path) -> dict:
    """Transpose sealed polygon intersections onto the exact terrain grid in Rust.

    Geometry only. A numerical producer must separately declare its spatial
    interpretation and resolve overlapping source polygons before physical use.
    """
    exe = _artifact(BINARY_ENV, "rw_hydrofabric.exe" if os.name == "nt" else "rw_hydrofabric")
    result = subprocess.run([str(exe), "prepare-cell-overlap", str(config)],
                            capture_output=True, text=True)
    if result.returncode:
        raise ValueError(result.stderr.strip() or "Rust cell-overlap preparation failed")
    return json.loads(result.stdout)


def parent_grid_contract(parent_geometry) -> dict:
    """Bind the driver's exact parent identity to Rust-derived weight-grid metadata.

    Accepts the forecast-scoped ParentGeometry, whose grid is the original
    ProjectedGrid, and whose restart_identity includes the definition and input
    authority hashes. This performs no coordinate or array arithmetic in Python.
    Pass result['grid'] as the preparation config's grid, and the whole result
    as parent_grid_contract. Rust validates the match and persists the contract;
    load_cuda retains it in the prepared weights' restart identity.
    """
    from woof.static import rust_bridge

    identity = parent_geometry.restart_identity
    if identity.get("schema") != "arwen.nextgen.parent-geometry.v1":
        raise ValueError("Hydrofabric parent grid requires the forecast ParentGeometry identity")
    if identity.get("definition") != parent_geometry.grid.definition():
        raise ValueError("Parent grid definition changed after the forecast geometry was bound")
    contract = json.loads(rust_bridge.grid_hydrofabric_json(
        parent_geometry.grid._rust_handle(rust_bridge)))
    if contract.get("schema") != "arwen.nextgen.parent-grid.v1":
        raise ValueError("Rust hydrofabric projected-grid contract schema mismatch")
    if contract["grid"]["row_order"] != identity.get("row_order"):
        raise ValueError("Rust parent grid row order differs from the forecast geometry identity")
    contract["parent_identity"] = identity
    return contract


def library_names():
    return ("nextgen_hydrofabric.dll" if os.name == "nt" else
                "libnextgen_hydrofabric.dylib" if sys.platform == "darwin" else
                "libnextgen_hydrofabric.so",)


def _library():
    lib = ctypes.CDLL(str(_artifact(LIBRARY_ENV, library_names()[0])))
    lib.arwen_hydrofabric_abi_version.argtypes = []
    lib.arwen_hydrofabric_abi_version.restype = ctypes.c_uint32
    if lib.arwen_hydrofabric_abi_version() != 1:
        raise RuntimeError("Rust hydrofabric startup ABI mismatch")
    lib.arwen_hydrofabric_read_into.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.arwen_hydrofabric_read_into.restype = ctypes.c_int
    return lib


class RawPayloadReader:
    """Production Rust raw reader, resolved once during component setup.

    Uses the standard packaged/staged bridge ladder and ABI1. The caller owns
    pinned allocation, dtype/shape metadata and H2D. Rust reads, checks exact
    length and SHA256, then copies bytes without interpreting their numbers.
    This bridge does not depend on the offline replay tool or a source tree.
    """

    def __init__(self):
        self._lib = _library()

    def read_into(self, path, expected_sha256: str, destination: int, nbytes: int) -> None:
        """Fill caller-owned writable memory; failure leaves it unchanged.

        destination is a native address valid for nbytes bytes through this
        synchronous call. Pass the manifest's expected digest, not a digest
        computed from the same untrusted payload. No allocation or H2D occurs.
        """
        if nbytes < 0 or (not destination and nbytes):
            raise ValueError("Rust payload transport requires valid writable memory")
        code = self._lib.arwen_hydrofabric_read_into(
            os.fsencode(path), expected_sha256.encode("ascii"), destination, nbytes)
        if code:
            raise RuntimeError(f"Rust payload length/hash/read failed for {path} (status {code})")

    def read_json(self, path, expected_sha256: str):
        """Read/hash metadata in Rust before Python parses the verified JSON."""
        size = Path(path).stat().st_size
        memory = ctypes.create_string_buffer(max(size, 1))
        self.read_into(path, expected_sha256, ctypes.addressof(memory), size)
        return json.loads(memory.raw[:size])


def read_raw_into(path, expected_sha256: str, destination: int, nbytes: int,
                  *, reader: RawPayloadReader | None = None) -> None:
    """Convenience setup call; reuse RawPayloadReader for multiple fields."""
    (reader if reader is not None else RawPayloadReader()).read_into(
        path, expected_sha256, destination, nbytes)


@dataclass
class PreparedHydrofabric:
    arrays: dict
    restart_identity: dict
    node_ids: tuple[str, ...]
    catchment_ids: tuple[str, ...]
    level_offsets: tuple[int, ...]
    reverse_level_offsets: tuple[int, ...]
    manifest: dict


def load_cuda(manifest_path: str | Path, *, device_id: int, array_module=None) -> PreparedHydrofabric:
    """Read verified LE payloads in Rust directly into pinned transport, once at setup."""
    if sys.byteorder != "little":
        raise RuntimeError("Hydrofabric GPU payload transport currently requires a little-endian host")
    path = Path(manifest_path).resolve()
    metadata = json.loads(path.read_text())
    if metadata.get("schema") != SCHEMA:
        raise ValueError("Unsupported hydrofabric manifest schema")
    roster_hash = metadata.get("auxiliary", {}).get("network.json", {}).get("sha256")
    if not roster_hash:
        raise ValueError("Hydrofabric manifest must declare the network.json SHA256")
    reader = RawPayloadReader()
    roster = reader.read_json(path.parent / "network.json", roster_hash)
    if array_module is None:
        import cupy as array_module
    cp = array_module
    arrays = {}
    with cp.cuda.Device(device_id):
        for name, field in metadata["files"].items():
            shape = tuple(field["shape"])
            dtype = cp.dtype(field["dtype"])
            size = int(field["bytes"])
            if math.prod(shape) * dtype.itemsize != size:
                raise ValueError(f"Hydrofabric shape/dtype length mismatch: {name}")
            file = (path.parent / field["path"]).resolve()
            if not file.is_relative_to(path.parent):
                raise ValueError(f"Hydrofabric payload escapes prepared directory: {name}")
            pinned = cp.cuda.alloc_pinned_memory(max(size, 1))
            pointer = ctypes.addressof(ctypes.c_char.from_buffer(pinned))
            reader.read_into(file, field["sha256"], pointer, size)
            array = cp.empty(shape, dtype=dtype)
            if size:
                cp.cuda.runtime.memcpy(array.data.ptr, pointer, size, cp.cuda.runtime.memcpyHostToDevice)
            arrays[name] = array
    restart_identity = {
        "schema": SCHEMA, "source_sha256": metadata["input"]["sha256"],
        "request_sha256": metadata["request_sha256"],
        "payload_sha256": {k: v["sha256"] for k, v in metadata["files"].items()},
        "network_sha256": roster_hash,
        "node_order": metadata["node_order"], "upstream_order": metadata["upstream_order"],
        "numerics": metadata["numerics"],
    }
    if "parent_grid_contract" in metadata:
        restart_identity["parent_grid_contract"] = metadata["parent_grid_contract"]
    return PreparedHydrofabric(
        arrays=arrays,
        restart_identity=restart_identity,
        node_ids=tuple(roster["node_ids"]), catchment_ids=tuple(roster["catchment_ids"]),
        level_offsets=tuple(roster["level_offsets"]),
        reverse_level_offsets=tuple(roster["reverse_level_offsets"]), manifest=metadata,
    )


def prepare_resource(*, settings, options, state, physics, device_id, mode, components):
    """Prepare-only realization resource: upload once; routing owns launches."""
    return load_cuda(options["manifest"], device_id=device_id)


def prepare_resource_memory(*, settings, options, context, components):
    """Declare every prepared upload before device construction."""
    from .hydro.data_memory import fabric_memory
    return fabric_memory(settings=settings, options=options, context=context, components=components)


def register_cli(hydro_subparsers):
    parser = hydro_subparsers.add_parser("fabric", help="prepare a NextGen hydrofabric GeoPackage in Rust")
    parser.add_argument("config", type=Path, help="JSON with geopackage, output, data_license and optional grid/subset")
    parser.set_defaults(func=_command)


def _command(args):
    print(json.dumps(prepare(args.config), indent=2))
    return 0
