"""``WOOF_HEX_PAIR_ADD``: the plain form of the forecast's ordered pair add, kept out of the regional kernel set.

``fused`` (default) adds the saved mixing increment and the physics increment to a tendency in one launch
(``add_pair_inplace_f32``, ``CudaDryDycoreDriver._add_pair`` in cuda_driver.py); ``plain`` makes the same two
rounded sums in the same order as two ``add_inplace_f32`` launches. The words are the same either way.

THE BREAKAGE THE PLAIN FORM PREVENTS: the differentiable engine's kernel sets have a derivative row for
add_inplace_f32 and none for the fused form, so a plan the adjoint door recorded from --init was refused in its
window ("add_pair_inplace_f32: no derivative row and no rule", x1.40962, 2026-11-11); the door records with the
plain form (ad-doors.v1.json record_env).

THE BREAKAGE ITS PLACE PREVENTS: cuda_driver.py is one of the sources the regional kernel-set digest is taken over
(cuda_backend/regional_admission.py REGIONAL_KERNEL_SOURCES). The switch first lived there (doors commit 089db92aa),
which moved the digest off MINTED_KERNEL_SET_SHA256, so every admitted limited-area class lapsed and a plain
regional CUDA forecast was refused ("was minted at kernel set 3be7dc867d7fcd77... re-mint the class";
tests/test_regional_forecast_anchor.py::test_a_different_cull_of_the_same_region_is_refused, 2026-11-11). Here the
digested bytes stay as minted: the plain form is installed over the method at import time, and only when asked.
"""

from __future__ import annotations

import os
from typing import Any

PAIR_ADD_KNOB = "WOOF_HEX_PAIR_ADD"


def pair_add_fused(environ: Any = None) -> bool:
    """True for ``fused`` (the default), False for ``plain``; any other word is refused by name."""
    value = (os.environ if environ is None else environ).get(PAIR_ADD_KNOB, "fused").strip().lower()
    if value not in ("fused", "plain"):
        raise ValueError(f"{PAIR_ADD_KNOB} must be fused or plain, got {value!r}")
    return value == "fused"


def install(environ: Any = None) -> bool:
    """Put the switch over ``CudaDryDycoreDriver._add_pair`` (once). Returns True when it was installed now.

    The installed method reads the switch at every call and hands the fused form to the original method, so a
    process that installs it and runs fused launches exactly what it launched before.
    """
    pair_add_fused(environ)  # refuse a wrong word before anything is touched
    from . import cuda_driver

    cls = cuda_driver.CudaDryDycoreDriver
    if getattr(cls._add_pair, "_hex_pair_add_form", False):
        return False
    fused = cls._add_pair

    def _add_pair(self, target, first, second):
        if pair_add_fused():
            return fused(self, target, first, second)
        nlevels, nowners = cuda_driver._level_owner_shape(target, "target")
        if tuple(first.shape) != tuple(target.shape) or tuple(second.shape) != tuple(target.shape):
            raise ValueError("increment shape differs from target")
        # The plain form: the same two rounded sums in the same order, as two add_inplace_f32 launches
        # (target = (target + first) + second in both; add_pair_inplace_f32 only saves a launch).
        self._add(target, first)
        self._add(target, second)

    _add_pair._hex_pair_add_form = True
    _add_pair.__wrapped__ = fused
    cls._add_pair = _add_pair
    return True


def install_from_environment() -> None:
    """Called by ``woof/hex/__init__.py``: installs the switch only when the environment names a form other than fused."""
    if os.environ.get(PAIR_ADD_KNOB, "fused").strip().lower() != "fused":
        install()
