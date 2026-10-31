#!/usr/bin/env python3
"""Real-library load, pin authentication and bitwise pool roundtrip check."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from woof.hex.mpasphys.native_backend import NativePhysics

library = Path(sys.argv[1]).resolve()
with NativePhysics(library, test_only=True, pin_set='rrfs-44c0849') as native:
    expected = np.asfortranarray([[1.0, -0.0], [3.5, -4.25]], dtype=np.float32)
    native.put_field('state', 'capture_roundtrip', expected)
    actual = native.get_field('state', 'capture_roundtrip')
    count = int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32)))
    assert count == 0
    result = dict(library_sha256=hashlib.sha256(native.path.read_bytes()).hexdigest(),
                  source_pin=native.build['source_pin'], pin_set=native.pin_set,
                  entries=len(native.abi['entries']), abi_version=native.lib.native_abi_version(),
                  compared_words=expected.size, different_words=count,
                  scope='capture tooling only; no module replay or forecast identity claim')
print(json.dumps(result, indent=2))
