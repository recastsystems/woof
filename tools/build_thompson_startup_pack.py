"""Build the cold-start Thompson images with the existing AOT pack builder.

Run in each supported CuPy/NVRTC environment on a leased CUDA host. Capture
the production loaders without compiling, then build every shipped SM in
parallel CPU compiler processes. No forecast or numerical options change.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def bind_pack(output):
    """Bind include trees exactly as the existing retained-image cache does."""
    from woof import kernel_aot, kernel_pack
    from woof.kernel_dependencies import include_roots, include_tree_dependencies
    identity = kernel_aot.compiler_identity()
    shipped = kernel_pack.read(output, identity)
    snapshots = {}
    for row in shipped:
        row['effective_compile']['options'] = kernel_aot.portable_options(
            row['effective_compile']['options'])
        options = kernel_aot.local_options(row['options'])
        if include_roots(options):
            row['include_trees'] = include_tree_dependencies(options, snapshots)
    kernel_pack.write(output, identity, shipped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--recipes', type=Path, nargs='*', default=[],
                        help='additional real first-step request catalogs for this compiler')
    args = parser.parse_args()
    from cupy.cuda import compiler
    from woof import kernel_aot, kernel_pack, numerics
    from woof.core import kernels
    from woof.kernel_aot_build import build

    class Captured(BaseException):
        pass

    entries = []
    fields = kernel_aot._FIELDS
    def capture(*positional, **keywords):
        bound = dict(zip(fields, positional))
        bound.update(keywords)
        entries.append(dict(source=bound['source'],
            options=list(kernel_aot.effective_options(bound.get('options') or ())),
            **kernel_aot.request_metadata(bound)))
        raise Captured

    original = compiler._compile_with_cache_cuda
    previous_mode = numerics._mode
    compiler._compile_with_cache_cuda = capture
    try:
        for mode in ('strict', 'fast'):
            numerics._mode = mode
            for name in ('thompson', 'thompson_classic'):
                try:
                    kernels.load_module.__wrapped__(name)
                except Captured:
                    pass
                else:
                    raise RuntimeError(f'{name}: production loader was not captured')
    finally:
        compiler._compile_with_cache_cuda = original
        numerics._mode = previous_mode
    kernel_pack.write(args.catalog, kernel_aot.compiler_identity(), entries)
    result = build(args.output, [args.catalog, *args.recipes])
    # Image inputs already use portable options. Normalize the diagnostic
    # copy too so package data contains no build-host installation paths.
    bind_pack(args.output)
    print(json.dumps(dict(result, compiler=kernel_aot.compiler_identity()), sort_keys=True))


if __name__ == '__main__':
    main()
