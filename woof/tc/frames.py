"""Orchestration of Rust canonical frame publication after a CUDA transform."""
import ctypes as C
import json
from pathlib import Path


class _Field(C.Structure):
    _fields_=[("name",C.c_char_p),("values",C.c_void_p),("count",C.c_size_t)]


def publish(*, library, input_directory, output_directory, valid_time,
            input_manifest_sha256, transform_sha256, fields):
    """Publish selected resident fields, with unchanged geometry and time roster.

    Numerical arrays return from CUDA once at this storage boundary. Rust
    verifies donor hashes, writes the stream in parallel, and publishes the
    manifest last. Output must be a new directory. Failed output has no valid
    manifest and must not be consumed. Provenance names the sealed transform.
    """
    import cupy as cp
    if not fields:
        raise ValueError("canonical TC publication requires updated fields")
    host={}
    for name,value in fields.items():
        if not isinstance(name,str) or not name or "\0" in name:
            raise ValueError("invalid canonical field name")
        if not isinstance(value,cp.ndarray) or value.dtype!=cp.float64 or not value.flags.c_contiguous:
            raise ValueError("canonical TC fields must be contiguous CUDA FP64")
        host[name]=cp.asnumpy(value)
    return publish_host(library=library,input_directory=input_directory,
                        output_directory=output_directory,valid_time=valid_time,
                        input_manifest_sha256=input_manifest_sha256,
                        transform_sha256=transform_sha256,fields=host)


def publish_host(*, library, input_directory, output_directory, valid_time,
                 input_manifest_sha256, transform_sha256, fields):
    """Store already computed fields through Rust, without numerical work.

    This storage boundary also permits a verified candidate to be transferred
    onto an identical background with additional surface inventory. Callers
    must establish that identity and bind it into the transform declaration.
    """
    import numpy as np
    if not fields:
        raise ValueError("canonical TC publication requires updated fields")
    host=dict(fields)
    for name,value in host.items():
        if not isinstance(name,str) or not name or "\0" in name:
            raise ValueError("invalid canonical field name")
        if not isinstance(value,np.ndarray) or value.dtype!=np.dtype('<f8') or not value.flags.c_contiguous:
            raise ValueError("canonical TC storage fields must be contiguous little-endian FP64")
    bindings=(_Field*len(host))()
    names=[]
    for i,(name,value) in enumerate(host.items()):
        names.append(name.encode("utf-8"))
        bindings[i]=_Field(names[-1],value.ctypes.data,value.size)
    request=json.dumps({"schema":"gpuwm-tc-frame-patch/1",
                       "input":str(Path(input_directory).resolve(strict=True)),
                       "output":str(Path(output_directory).resolve()),
                       "valid_time":valid_time,"input_manifest_sha256":input_manifest_sha256,
                       "transform_sha256":transform_sha256},
                      sort_keys=True,allow_nan=False).encode()
    lib=C.CDLL(str(Path(library).resolve(strict=True)))
    lib.tc_frames_abi_version.restype=C.c_uint32
    if lib.tc_frames_abi_version() not in (1,2,3,4):
        raise RuntimeError("unsupported canonical TC writer ABI")
    lib.tc_frames_publish.argtypes=[C.c_char_p,C.POINTER(_Field),C.c_size_t]
    lib.tc_frames_publish.restype=C.c_int
    lib.tc_frames_last_error.restype=C.c_char_p
    if lib.tc_frames_publish(request,bindings,len(bindings)):
        raise RuntimeError(lib.tc_frames_last_error().decode())
    return Path(output_directory)/"frames.json"


def read(*, library, input_directory, valid_time, fields, device=True):
    """Read selected full fields through parallel Rust checks, then upload.

    Returns device arrays, the selected frame descriptor and the exact manifest
    digest to bind publication. NumPy allocates buffers only; Rust owns raw I/O.
    """
    import hashlib
    import numpy as np
    directory=Path(input_directory).resolve(strict=True)
    manifest=(directory/'frames.json').read_bytes()
    digest=hashlib.sha256(manifest).hexdigest();doc=json.loads(manifest)
    selected=[f for f in doc['frames'] if f['valid_time']==valid_time]
    if len(selected)!=1 or 'atmospheric_window' in selected[0]:
        raise ValueError('TC transform requires exactly one full initial time/member')
    frame=selected[0];names=tuple(fields)
    if not names or len(set(names))!=len(names):
        raise ValueError('TC frame reader needs unique field names')
    host={}
    for name in names:
        matches=[f for f in frame['fields'] if f['name']==name]
        if len(matches)!=1 or matches[0]['dtype']!='<f8':
            raise ValueError(f'canonical field absent, duplicated or incompatible: {name}')
        host[name]=np.empty(tuple(matches[0]['shape']),dtype='<f8')
    bindings=(_Field*len(host))();encoded=[]
    for i,(name,value) in enumerate(host.items()):
        encoded.append(name.encode());bindings[i]=_Field(encoded[-1],value.ctypes.data,value.size)
    lib=C.CDLL(str(Path(library).resolve(strict=True)))
    lib.tc_frames_abi_version.restype=C.c_uint32
    if lib.tc_frames_abi_version() not in (2,3,4):
        raise RuntimeError('canonical TC reader needs frame I/O ABI 2 or 3')
    lib.tc_frames_read.argtypes=[C.c_char_p,C.POINTER(_Field),C.c_size_t]
    lib.tc_frames_read.restype=C.c_int;lib.tc_frames_last_error.restype=C.c_char_p
    request=json.dumps({'schema':'gpuwm-tc-frame-read/1','input':str(directory),
        'valid_time':valid_time,'input_manifest_sha256':digest},allow_nan=False).encode()
    if lib.tc_frames_read(request,bindings,len(bindings)):
        raise RuntimeError(lib.tc_frames_last_error().decode())
    if not device:return host,frame,digest
    import cupy as cp
    return {name:cp.asarray(value) for name,value in host.items()},frame,digest
