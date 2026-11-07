"""Persistent direct-NVRTC images, without CuPy's RawModule FTZ override."""

from woof.backend import compilation_target as _compilation_target
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import uuid

from ..cuda_backend.compile_contract import (NvrtcCompileCapture,
    compile_platform_binding, canonical_sha256)


@lru_cache(maxsize=None)
def _platform(device):
    # Loaded compiler/driver libraries are fixed for this process. The shared
    # measured binding includes NVRTC build and library digest, not just ABI.
    return compile_platform_binding()


def compile_image(source, *, options, arch=None, cache_dir=None):
    """Return the image and its captured effective compile evidence.

    Disk hits require exact source, options, architecture and compiler identity
    plus the image digest. Evidence belongs to the original compilation; a hit
    explicitly reports that origin rather than claiming NVRTC ran again.
    """
    import cupy as cp
    from cupy.cuda import compiler
    device = cp.cuda.runtime.getDevice()
    arch = arch or _compilation_target(device, xp=cp)
    source_hash = hashlib.sha256(source.encode('utf-8')).hexdigest()
    identity = dict(schema=1, source_sha256=source_hash, options=list(options),
                    arch=str(arch), platform=_platform(device))
    key = canonical_sha256(identity)
    root = Path(cache_dir or os.environ.get('CUPY_CACHE_DIR') or compiler.get_cache_dir()) / 'mpas-strict-v842'
    image_path, record_path = root / (key + '.bin'), root / (key + '.json')
    try:
        record = json.loads(record_path.read_text())
        image = image_path.read_bytes()
        evidence = record['effective_compile']
        if (record['identity'] == identity and isinstance(evidence, dict)
                and record['image_sha256'] == hashlib.sha256(image).hexdigest()):
            return image, {**evidence, 'persistent_cache_hit': True,
                           'persistent_image_sha256': record['image_sha256']}
    except (OSError, ValueError, KeyError, TypeError):
        pass
    with NvrtcCompileCapture() as capture:
        image, _ = compiler.compile_using_nvrtc(source, options=tuple(options),
                                                arch=arch, cache_in_memory=True)
    if isinstance(image, str):
        image = image.encode('utf-8')
    evidence = capture.evidence(source_hash)
    digest = hashlib.sha256(image).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    suffix = '.' + uuid.uuid4().hex
    image_tmp = image_path.with_name(image_path.name + suffix)
    record_tmp = record_path.with_name(record_path.name + suffix)
    image_tmp.write_bytes(image)
    record_tmp.write_text(json.dumps(dict(identity=identity, image_sha256=digest,
        effective_compile=evidence), sort_keys=True) + '\n')
    os.replace(image_tmp, image_path)
    os.replace(record_tmp, record_path)
    return image, {**evidence, 'persistent_cache_hit': False,
                   'persistent_image_sha256': digest}
