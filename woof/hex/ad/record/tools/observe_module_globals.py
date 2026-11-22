#!/usr/bin/env python3
"""Observe module globals and direct NVRTC source bindings in a forward run.

Addresses are observations for matching only: a replay must query the symbol
from its own live module. This does not establish a carried-state census.
"""
# SPDX-License-Identifier: Apache-2.0
import functools
import gzip
import hashlib
import json
import sys
import threading
import time

import observe_compiler_sources as sources
import observe_cupti_outer as base
import observe_seaice_pointers as seaice


def image_bytes(value):
    if isinstance(value, str):
        return value.encode('utf-8')
    if isinstance(value, bytes):
        return value
    raise TypeError('NVRTC image is not bytes or text')


def global_binding(module, name, result, clock_ns, thread_id):
    if not isinstance(name, str) or not name or '\0' in name:
        raise ValueError('invalid module-global name')
    image = getattr(module, 'image', None)
    if not isinstance(image, str) or len(image) != 64:
        raise ValueError('module-global has no observed module image')
    int(image, 16)
    address, size = (result[0], result[1] if len(result) > 1 else None) if isinstance(result, tuple) else (result, None)
    address, pointer = int(address), int(module.ptr)
    if not 0 < address < (1 << 64) or pointer <= 0:
        raise ValueError('invalid module-global address or module')
    if size is not None:
        size = int(size)
        if size <= 0:
            raise ValueError('invalid module-global size')
    return {'schema': 'ad.binder.phys-module-global/v1', 'complete': False,
            'clock_ns': clock_ns, 'thread_id': thread_id, 'module': pointer,
            'image_sha256': image, 'name': name, 'address': address, 'bytes': size}


class GlobalObserver(sources.SourceObserver):
    def __init__(self, root):
        super().__init__(root)
        self.global_output = gzip.open(root / 'module-globals.jsonl.gz', 'wt',
                                       encoding='utf-8', compresslevel=1)
        self.direct_sources = {}

    def install(self):
        super().install()
        from cupy.cuda import function
        from woof import nvrtc_ptx_cache
        observer = self
        original_module = function.Module
        original_compile = nvrtc_ptx_cache.compile_using_nvrtc

        @functools.wraps(original_compile)
        def observe_direct_compile(source, *args, **kwargs):
            result = original_compile(source, *args, **kwargs)
            image = image_bytes(result[0])
            if not isinstance(source, str):
                raise TypeError('NVRTC source is not text')
            digest = hashlib.sha256(image).hexdigest()
            with observer.lock:
                observer.direct_sources.setdefault(digest, set()).add(source)
            return result

        class Module(original_module):
            def load(module, image):
                result = super().load(image)
                digest = hashlib.sha256(image_bytes(image)).hexdigest()
                with observer.lock:
                    matching = tuple(observer.direct_sources.get(digest, ()))
                for source in matching:
                    observer.observe_source(source, module)
                return result

            def get_global_var(module, name):
                result = super().get_global_var(name)
                row = global_binding(module, name, result, time.monotonic_ns(),
                                     threading.get_native_id())
                with observer.lock:
                    observer.global_output.write(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')
                    observer.global_output.flush()
                return result

        nvrtc_ptx_cache.compile_using_nvrtc = observe_direct_compile
        function.Module = Module

    def close(self):
        try:
            self.global_output.close()
        finally:
            super().close()


if __name__ == '__main__':
    base.Observer = GlobalObserver
    base.boundary_trace = seaice.boundary_species_and_seaice_trace
    sys.exit(base.main())
