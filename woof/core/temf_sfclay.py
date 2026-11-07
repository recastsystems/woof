"""TEMF surface layer, ``sf_sfclay_physics = 10``.

CUDA launcher around :mod:`woof.core.kernels.temf_sfclay`, the transcription
of WRF v4.6.1 ``phys/module_sf_temfsfclay.F`` (``TEMFSFCLAY1D``).  The scheme
is a lowest-level column formula: one thread per column, no tables, no
iteration.  It shares two pieces of carried state with the TEMF PBL
(``bl_pbl_physics = 10``) and owns one:

* ``TE_TEMF`` (first level) is the PBL's total energy; the surface layer
  floors it at ``TEmin`` in place (module_sf_temfsfclay.F:256);
* ``HD_TEMF`` is the PBL's dry thermal top, read for the w* velocity scale;
* ``WM_TEMF`` is this scheme's own velocity scale, averaged with its
  previous value every call (:338-340) and carried across restarts.

WRF pairs the two schemes both ways (share/module_check_a_mundo.F:2287-2299);
:func:`woof.config.validate_temf_pairing` is that law.

Graded at 0 ULP against tools/temf_sfclay_wrf461_oracle by
tests/test_temf_sfclay_wrf461_parity.py.
"""

from __future__ import annotations

import numpy as np

from woof.core.kernels import get_kernel
from woof.core.physics_inventory import (TEMF_SFCLAY_CARRIED,
                                          TEMF_SFCLAY_INPUTS,
                                          TEMF_SFCLAY_COLUMN_OUTPUTS)
from woof.core.state import DTYPE

_TPB = 128


def launch_temf_sfclay(inputs, carried, outputs) -> None:
    """Run one TEMF surface-layer step in place.

    ``inputs`` holds the read-only lowest-level and surface fields
    (:data:`TEMF_SFCLAY_INPUTS`), ``carried`` the three fields the scheme
    reads and rewrites (:data:`TEMF_SFCLAY_CARRIED`: the first level of
    TE_TEMF, HFX and WM_TEMF), ``outputs`` the fields it writes
    (:data:`TEMF_SFCLAY_COLUMN_OUTPUTS`).  Every array is contiguous float32 of
    one surface shape; ``carried["te1"]`` may be a contiguous view of the
    PBL's TE_TEMF bottom level, which is updated in place.
    """
    shape = inputs["psfc"].shape
    named = ([inputs[name] for name in TEMF_SFCLAY_INPUTS]
             + [carried[name] for name in TEMF_SFCLAY_CARRIED]
             + [outputs[name] for name in TEMF_SFCLAY_COLUMN_OUTPUTS])
    for array in named:
        if (array.shape != shape or array.dtype != DTYPE
                or not array.flags.c_contiguous):
            raise ValueError(
                "launch_temf_sfclay requires same-shape contiguous float32 "
                f"surface arrays of shape {shape}")
    n = int(np.prod(shape))
    blocks = (n + _TPB - 1) // _TPB
    kernel = get_kernel("temf_sfclay", "temf_sfclay_column")
    kernel((blocks,), (_TPB,), tuple(named) + (np.int32(n),))


__all__ = ["launch_temf_sfclay"]
