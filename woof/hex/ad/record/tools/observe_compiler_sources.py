#!/usr/bin/env python3
"""Observe source-to-module bindings even when CuPy's binary cache hits.

This is an ordinary-forward observation only.  Its records are not replay
inputs and do not establish kernel arguments or a carried-state census.
"""
# SPDX-License-Identifier: Apache-2.0
import functools
import gzip
import hashlib
import json
import sys
import threading
import time

import observe_cupti_outer as base
import observe_seaice_pointers as seaice


def source_binding(source, module, clock_ns, thread_id):
    """Return source bytes and module identity, or None for non-module output."""
    if not isinstance(source, str):
        raise TypeError('compiler source is not text')
    if not hasattr(module, 'ptr'):
        return None
    pointer = int(module.ptr)
    image = getattr(module, 'image', None)
    if pointer <= 0 or not isinstance(image, str) or len(image) != 64:
        raise ValueError('compiled module lacks an observed image binding')
    int(image, 16)
    data = source.encode('utf-8')
    return data, {'schema': 'ad.binder.phys-source-binding/v1', 'complete': False,
                  'clock_ns': clock_ns, 'thread_id': thread_id,
                  'module': pointer, 'image_sha256': image,
                  'source_sha256': hashlib.sha256(data).hexdigest(),
                  'source_bytes': len(data)}


class SourceObserver(seaice.SeaiceObserver):
    def __init__(self, root):
        super().__init__(root)
        self.sources = root / 'sources'
        self.sources.mkdir(exist_ok=True)
        self.source_output = gzip.open(root / 'source-bindings.jsonl.gz', 'wt',
                                       encoding='utf-8', compresslevel=1)

    def observe_source(self, source, module):
        binding = source_binding(source, module, time.monotonic_ns(), threading.get_native_id())
        if binding is None:
            return
        data, row = binding
        with self.lock:
            target = self.sources / (row['source_sha256'] + '.cu.gz')
            if not target.exists():
                with gzip.open(target, 'wb', compresslevel=1) as output:
                    output.write(data)
            self.source_output.write(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')
            self.source_output.flush()

    def install(self):
        super().install()
        from cupy.cuda import compiler
        original = compiler._compile_with_cache_cuda

        @functools.wraps(original)
        def observe_compile(source, *args, **kwargs):
            module = original(source, *args, **kwargs)
            self.observe_source(source, module)
            return module

        compiler._compile_with_cache_cuda = observe_compile

    def close(self):
        try:
            self.source_output.close()
        finally:
            super().close()


if __name__ == '__main__':
    base.Observer = SourceObserver
    base.boundary_trace = seaice.boundary_species_and_seaice_trace
    sys.exit(base.main())
