"""Native advance_uv reconstructs spacing from its stored REAL reciprocal."""
from types import SimpleNamespace

import numpy as np


def test_native_one_km_damping_scale(monkeypatch):
    from woof.core import acoustic

    monkeypatch.setattr(acoustic, "WRF_EXACT", True)
    state = SimpleNamespace(mup=object(), msfu=object(), msfv_inv=object(),
                            has_msf=True)
    cfg = SimpleNamespace(dx=1000.0, dy=3333.3333333333335, emdiv=0.01)
    mudf = object()
    args = acoustic._uv_emdiv_args(state, cfg, mudf)
    # GNU 13.3, strict WRF 4.6.1 advance_uv's REAL dx=1./rdx:
    # dx=999.99993896484375, -emdiv*dx=-9.99999904632568359375.
    assert args[3].view(np.uint32) == 0xC11FFFFF
    assert args[4].view(np.uint32) == 0xC2055555
    assert args[0] is mudf
    assert int(args[-2]) == 1


def test_fast_mode_keeps_existing_kernel_arguments(monkeypatch):
    from woof.core import acoustic

    monkeypatch.setattr(acoustic, "WRF_EXACT", False)
    assert acoustic._uv_emdiv_args(None, None, None) == ()
