#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""P1 sealer-output-steps: write the isobaric diagnostics' CUDA source (hexcore mpasdiag/_artifact.cuda_source(),
the exact text mpasdiag/isobaric.py:_module compiles) for stage 2's --layout-source, with a receipt of what ties it
to the recorded image.

Why: the module is loaded by cp.cuda.function.Module().load(bytes) at the 3-hourly output steps only, so the
source observer never saw it, and the 18-step seal left all 92 arguments of its launches (prepare_mass,
prepare_interfaces, prepare_vertices, slp_columns, slp_finish, layer_mean at steps 9 and 18) unresolved.

What ties source and image (all checked here): the image's kernel-name set equals the source's __global__ set, and
a recompile with the recording's toolkit (wheel NVRTC 13.4, strict options) gives an image of the recorded length
less the trailing NUL whose bytes differ only from the ELF compiler-options note on (byte offset recorded below).
Stage 2 additionally requires every launch's parameter count and widths to equal the parsed signature, and
every pointer to resolve to one live allocation.
Usage: make_layout_source.py RECORDED_IMAGE.gz OUT_DIR   (run with the recording's PYTHONPATH and CUDA_PATH)
"""
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path


def main():
    image_path, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    from woof.hex import numerics
    from woof.hex.mpasdiag import _artifact
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from source_signatures import image_signatures
    source = _artifact.cuda_source().encode()
    recorded = gzip.decompress(image_path.read_bytes())
    names = set(n.decode() for n in re.findall(rb'\.text\.([A-Za-z_][A-Za-z_0-9]*)\x00', recorded))
    sigs = image_signatures(source)
    receipt = {'schema': 'ad.binder.phys-layout-source/v1', 'source_sha256': hashlib.sha256(source).hexdigest(),
               'recorded_image_sha256': hashlib.sha256(recorded).hexdigest(), 'recorded_bytes': len(recorded),
               'image_kernels': sorted(names), 'source_kernels': sorted(sigs),
               'kernel_sets_equal': names == set(sigs),
               'signatures': {k: v for k, v in sorted(sigs.items()) if k in names}}
    try:
        import cupy as cp
        from cupy.cuda import compiler
        binary, _ = compiler.compile_using_nvrtc(source.decode(), options=numerics.options(_artifact.OPTIONS, numerics='strict'),
                                                 arch=cp.cuda.Device(0).compute_capability)
        n = min(len(binary), len(recorded))
        first = next((k for k in range(n) if binary[k] != recorded[k]), n)
        note = recorded.find(b'NVIDIA Corp')
        receipt['recompile'] = {'bytes': len(binary), 'equal_plus_nul': recorded == binary + b'\0',
                                'first_differing_byte': first, 'recorded_options_note': recorded[first - 4:first + 40].decode('latin-1'),
                                'recompiled_options_note': binary[first - 4:first + 40].decode('latin-1'),
                                'nvidia_note_offset': note}
    except Exception as error:  # the receipt says so; stage 2's checks do not depend on it
        receipt['recompile'] = {'error': repr(error)}
    (out / 'mpasdiag-isobaric.cu.gz').write_bytes(gzip.compress(source, 1, mtime=0))
    (out / 'mpasdiag-isobaric-layout.json').write_text(json.dumps(receipt, indent=1) + '\n')
    print(json.dumps({k: receipt[k] for k in ('kernel_sets_equal', 'source_sha256', 'recorded_image_sha256')}),
          json.dumps(receipt['recompile'])[:400])
    if not receipt['kernel_sets_equal']:
        raise SystemExit('image and source kernel sets differ')


if __name__ == '__main__':
    main()
