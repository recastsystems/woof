"""Setup-only orchestration for the Rust catchment forcing engine.

Metadata is Python; external arrays, NetCDF/CSV decoding, unit conversion,
regridding, hash verification and packet reads are Rust. CUDA owns stepping.
"""
from __future__ import annotations
import ctypes
import json
import os
from pathlib import Path
import sys

ARTIFACT = "arwen_nextgen_forcing"
CRATE = "tools/nextgen_forcing"
ENV = "WOOF_NEXTGEN_FORCING_BRIDGE"
ABI_MARKER = b"arwen_forcing_read_packet"


def library_names():
    return (ARTIFACT + ".dll" if sys.platform == "win32" else
            "lib" + ARTIFACT + (".dylib" if sys.platform == "darwin" else ".so"),)


def _library(path=None):
    if path is None:
        from woof import bridges
        filename = library_names()[0]
        path=bridges.find_artifact(ENV,filename,crate_relative=CRATE,executable=False)
        # On demand since WOOF 1.1.6 (PyPI per-file limit): staged by the first hydro
        # preparation from this release's bundle (never inside inspection_only()).
        if path is None and not os.environ.get(ENV) and bridges.fetch_on_demand(ARTIFACT):
            path=bridges.find_artifact(ENV,filename,crate_relative=CRATE,executable=False)
        if path is None:
            raise RuntimeError(bridges.artifact_remedy(env_var=ENV,filename=filename,
                subject="NextGen forcing Rust engine",crate_relative=CRATE,artifact=ARTIFACT))
    lib=ctypes.CDLL(str(path))
    lib.arwen_forcing_abi_version.restype=ctypes.c_uint32
    if lib.arwen_forcing_abi_version()!=1:
        raise RuntimeError("NextGen forcing Rust ABI mismatch")
    return lib


def prepare_external(request_path, *, library=None):
    """Prepare the explicit JSON request using its shared source-table row."""
    lib=_library(library)
    fn=lib.arwen_forcing_prepare
    fn.argtypes=[ctypes.c_char_p];fn.restype=ctypes.c_int
    request_path=Path(request_path).resolve()
    request=json.loads(request_path.read_text())
    if "source" in request:
        from .forcing import source_profile
        if "source_profile" in request or "profile_row" in request:
            raise ValueError("select either an init source or an explicit profile row")
        from woof.source_adapters import get_source_adapter
        source=request.pop("source")
        profile,row=source_profile(source)
        request["source_id"]=get_source_adapter(source).source_id
        request["source_profile"],request["profile_row"]=str(profile),row
        output=Path(request["output"]).resolve()
        output.mkdir(parents=True,exist_ok=True)
        request_path=output/"request.json"
        request_path.write_text(json.dumps(request,indent=2)+"\n")
    if fn(bytes(request_path)):
        raise RuntimeError("Rust forcing preparation failed; see its input-specific diagnostic")
    return Path(request["output"])/"manifest.json"


def load_series(manifest_path, *, library=None, device_id=0, native_precision=False,
                start_seconds=0.0):
    """Upload a hash-checked Rust packet and bind stable catchment views once."""
    import cupy as cp
    import numpy as np
    from .forcing import PreparedForcingSeries
    manifest_path=Path(manifest_path)
    metadata=json.loads(manifest_path.read_text())
    if metadata.get("schema")!="arwen.nextgen-forcing.v1":
        raise ValueError("unsupported Rust forcing packet schema")
    shape=tuple(metadata["shape"])
    if len(shape)!=3 or shape[1]!=8 or any(type(v) is not int or v<=0 for v in shape):
        raise ValueError("invalid Rust forcing packet shape")
    filename="forcing.f64" if native_precision else "forcing.f32"
    dtype=metadata["files"][filename]
    if dtype!=("<f8" if native_precision else "<f4"):
        raise ValueError("Rust forcing packet dtype differs from selected precision")
    count=shape[0]*shape[1]*shape[2]
    size=count*(8 if native_precision else 4)
    storage=(ctypes.c_ubyte*size)()
    lib=_library(library)
    fn=lib.arwen_forcing_read_packet
    fn.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.POINTER(ctypes.c_ubyte),ctypes.c_size_t]
    fn.restype=ctypes.c_int
    if fn(bytes(manifest_path.parent/filename),metadata["data_sha256"][filename].encode(),storage,size):
        raise RuntimeError("Rust forcing packet bytes/hash differ from preparation manifest")
    # A zero-copy typed view of Rust-filled storage; no host transformation.
    host=np.ndarray((count,),dtype=dtype,buffer=storage)
    with cp.cuda.Device(device_id):
        frames=cp.asarray(host).reshape(shape)
    return PreparedForcingSeries(frames,interval_seconds=metadata["interval_seconds"],
        time_origin=metadata["time_origin"],start_seconds=start_seconds,
        identity={"packet_sha256":metadata["data_sha256"][filename],
                  "catchment_ids":metadata["catchment_ids"],
                  "source_profile_sha256":metadata["source_profile_sha256"],
                  "source_id":metadata.get("source_id"),"profile_row":metadata["profile_row"],
                  "area_weights_sha256":metadata["area_weights_sha256"]})


def prepare_series(*, settings, options, state, physics, device_id, mode, components):
    """Realization factory: a previously prepared, hash-bound Rust packet."""
    if mode != "standalone":
        raise ValueError("external forcing series requires standalone hydro mode")
    if options.get("source"):
        import hashlib
        from .forcing import source_profile
        from woof.source_adapters import get_source_adapter
        table,row=source_profile(options["source"])
        source_id=get_source_adapter(options["source"]).source_id
        packet=json.loads(Path(options["manifest"]).read_text())
        if packet.get("source_id")!=source_id or packet["profile_row"]!=row or packet["source_profile_sha256"]!=hashlib.sha256(table.read_bytes()).hexdigest():
            raise ValueError("forcing packet profile differs from the selected init-source table")
    component=load_series(options["manifest"], library=options.get("library"),
                          device_id=device_id,
                          native_precision=options.get("native_precision", False),
                          start_seconds=options.get("start_seconds", 0.0))
    if options.get("time_origin") and component.restart_identity["time_origin"]!=options["time_origin"]:
        raise ValueError("forcing packet epoch differs from the realization")
    return component


def prepare_series_memory(*, settings, options, context, components):
    from .data_memory import series_memory
    return series_memory(settings=settings, options=options, context=context, components=components)


def register_cli(hydro_subparsers):
    parser=hydro_subparsers.add_parser("forcing",help="prepare table-driven NextGen forcing in Rust")
    parser.add_argument("request",type=Path,help="JSON naming source or explicit source_profile/profile_row")
    parser.set_defaults(func=_command)


def _command(args):
    print(prepare_external(args.request))
    return 0
