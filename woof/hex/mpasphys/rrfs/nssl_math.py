"""NSSL shared device maths, using the shipped permissive sources."""
from functools import lru_cache
from pathlib import Path

@lru_cache(maxsize=1)
def source():
    from ..resources_build import _vendored_maths, mynn_maths
    shared = mynn_maths(_vendored_maths())
    aliases = []
    for name in ("exp", "log", "pow"):
        for kind in ("float", "double"):
            parameters = f"{kind} a, {kind} b" if name == "pow" else f"{kind} a"
            arguments = "a,b" if name == "pow" else "a"
            aliases.append(f"__device__ {kind} nssl_{name}({parameters}) {{ return mynn_{name}({arguments}); }}")
    for name in ("log10", "atan", "tanh"):
        aliases.append(f"__device__ float nssl_{name}(float a) {{ return mynn_{name}(a); }}")
    # CUDA's binary64 sine, as in the shared TEMPO math provider. No LGPL
    # implementation enters the wheel. Strict claims name the real replay scope.
    aliases.append("__device__ double nssl_sin(double a) { return sin(a); }")
    return shared + "\n" + Path(__file__).with_name("nssl_sunpro.cuh").read_text() + "\n" + "\n".join(aliases)
