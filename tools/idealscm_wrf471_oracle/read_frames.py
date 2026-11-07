"""Read both production IDEALSCMSFCLAY CUDA entry points, in this mode."""
import json
import os

import cupy as cp

from woof.certify.compile_platform import compile_platform_fingerprint
from woof.core.idealscmsfclay import _module
from woof.core.kernels import kernel_local_size_bytes

module = _module(cp.cuda.Device().id)
print(json.dumps({
    "mode": "strict" if os.environ.get("GPUWM_WRF_EXACT") == "1" else "fast",
    "platform": compile_platform_fingerprint(),
    "frames": {name: kernel_local_size_bytes(module.get_function(name))
               for name in ("ideal_scm_advance", "ideal_scm_surface")},
}, indent=2), flush=True)
