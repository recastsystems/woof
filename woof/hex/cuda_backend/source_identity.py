"""Authenticate legacy MYNN packets after a preprocessor-only HIP guard.

Published CUDA packets pin the source before its AMD register-hint guard.
The exact inverse edit below reconstructs those pinned bytes; it cannot
normalize arithmetic, whitespace or any other source change.
"""
from hashlib import sha256


def mynn_source_matches(source: str, expected: str) -> bool:
    if sha256(source.encode()).hexdigest() == expected:
        return True
    guarded = '#if defined(HEX_NUMERICS_FAST) && !defined(__HIP_PLATFORM_AMD__)\n#define MYNN_L4_REGS __maxnreg__(160)'
    legacy = '#ifdef HEX_NUMERICS_FAST\n#define MYNN_L4_REGS __maxnreg__(160)'
    return source.count(guarded) == 1 and sha256(source.replace(guarded, legacy).encode()).hexdigest() == expected
