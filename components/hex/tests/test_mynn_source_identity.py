from hashlib import sha256
from woof.hex.cuda_backend.source_identity import mynn_source_matches


def test_packet_authentication_allows_only_the_exact_hip_guard():
    old = '#ifdef HEX_NUMERICS_FAST\n#define MYNN_L4_REGS __maxnreg__(160)\n#else\n#define MYNN_L4_REGS\n#endif\nx = a + b;'
    new = old.replace('#ifdef HEX_NUMERICS_FAST', '#if defined(HEX_NUMERICS_FAST) && !defined(__HIP_PLATFORM_AMD__)')
    pin = sha256(old.encode()).hexdigest()
    assert mynn_source_matches(new, pin)
    assert mynn_source_matches(new, sha256(new.encode()).hexdigest())
    assert not mynn_source_matches(new.replace('a + b', 'a - b'), pin)
    assert not mynn_source_matches(new + '\n', pin)
    assert not mynn_source_matches(new + new, pin)
