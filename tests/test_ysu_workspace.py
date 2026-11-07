"""The gates the YSU column workspace owns.

The breakage each one prevents, named:

1. ``test_the_ysu_frame_stays_under_the_default_stack`` -- the whole
   point, and the one that reached USERS.  YSU's column arrays used to
   live in the per-thread local frame, and CUDA sizes ONE per-context
   local-memory backing store to the widest frame it LAUNCHES times the
   card's RESIDENT-THREAD CAPACITY, not times the occupancy the kernel
   achieves.  MEASURED on test host 1 (RTX 5070 Ti, 70 SMs x 1,536,
   sm_120) through the real launcher at nz=49: a 9,232 B frame took 842.0
   MiB at launch and never gave it back.  ``bl_pbl_physics = 1`` is the
   wizard's default (woof/domain_wizard.py:714), so that was the widest
   frame a BARE DEFAULT run launched and every default run paid it.
   Anything that puts a column array back on the stack -- a new
   ``real x[YSU_KMAX]``, a compiler that stops promoting -- brings the
   reservation back, and this is the only assertion that catches it on a
   platform with no recorded row.

2. ``test_the_python_side_mirrors_the_kernels_workspace_geometry`` -- the
   launcher allocates the workspace and the kernel indexes it.  If the two
   disagree about slot count or lane width the kernel writes past the
   allocation.  ``YSUWS_LANES`` is the LAUNCH BLOCK: a launch at any other
   block width aliases lanes, silently, into a wrong forecast rather than
   a crash.

3. ``test_every_workspace_slot_id_is_used_once`` -- the slot ids are
   literal because NVRTC has no ``__COUNTER__``.  A duplicate id makes two
   column arrays alias, which is a wrong forecast and not a crash; an id
   past the declared count runs off the region.

4. ``test_the_workspace_is_free_of_residue`` (GPU) -- the workspace is
   allocated with ``cp.empty`` and reused across tiles, exactly as the
   local frame was reused across launches.  That is only sound if no
   column array is ever read before it is written.  This fills the
   workspace with two different garbage patterns and asserts the outputs
   are bit-identical.

5. ``test_tiling_does_not_change_the_answer`` (GPU) -- the columns are
   launched in tiles now.  A tile boundary that dropped or double-counted
   a column would be a wrong forecast in a strip of the domain.  This runs
   the same field through one tile and through many and asserts bitwise
   equality.
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from woof.core.ysu import (                                 # noqa: E402
    YSU_BLOCK, YSUWS_SLOTS, ysu_workspace_floats,
)

#: The CUDA default per-thread stack.  A frame at or under this reserves
#: nothing, because the driver's backing store is
#: ``(frame - default stack) x SMs x threads/SM`` floored at zero.
DEFAULT_STACK_BYTES = 1024


def _source() -> str:
    from woof.core.kernels import _unit_text
    return _unit_text("ysu")


def _define(name: str) -> int:
    m = re.search(rf"^#define {name} (\d+)$", _source(), re.M)
    assert m, f"the live YSU unit has no #define {name}"
    return int(m.group(1))


# ---------------------------------------------------------------------------
# source-only gates
# ---------------------------------------------------------------------------
def test_the_python_side_mirrors_the_kernels_workspace_geometry():
    """Launcher and kernel must agree, or the kernel writes out of bounds."""
    assert _define("YSUWS_SLOTS") == YSUWS_SLOTS
    assert _define("YSUWS_LANES") == YSU_BLOCK, (
        "YSUWS_LANES is the launch block width; ysu.py must launch at "
        "exactly that block or the lanes alias")


def test_every_workspace_slot_id_is_used_once():
    ids = [int(m) for m in re.findall(r"YSUWS_AT\(wsb,\s*(\d+),", _source())]
    assert ids, "no workspace slots found in the live YSU unit"
    assert len(ids) == YSUWS_SLOTS, (
        f"the live YSU unit binds {len(ids)} slots but declares YSUWS_SLOTS="
        f"{YSUWS_SLOTS}")
    assert sorted(ids) == list(range(YSUWS_SLOTS)), (
        f"slot ids must be 0..{YSUWS_SLOTS - 1} exactly once, got "
        f"{sorted(ids)}")


def test_no_column_array_is_left_on_the_stack():
    """A `real name[YSU_KMAX]` in the kernel body is the regression."""
    body = _source().split("void ysu_column(", 1)[1]
    left = re.findall(r"\breal\s+\w+\s*\[\s*YSU_KMAX", body)
    assert not left, (
        f"these column arrays are back in the local frame: {left}; that "
        "re-arms the per-context local-memory reservation this workspace "
        "exists to remove")


def test_the_workspace_grows_with_levels_not_with_the_kernels_bound():
    """The extent is a runtime argument, so nz is what it follows."""
    a = ysu_workspace_floats(49, YSU_BLOCK)
    b = ysu_workspace_floats(98, YSU_BLOCK)
    assert b > a
    assert a == YSUWS_SLOTS * 50 * YSU_BLOCK
    assert b == YSUWS_SLOTS * 99 * YSU_BLOCK


# ---------------------------------------------------------------------------
# device gates
# ---------------------------------------------------------------------------
@pytest.mark.gpu
def test_the_ysu_frame_stays_under_the_default_stack():
    """The reservation this workspace exists to remove, on ANY card."""
    import cupy  # noqa: F401
    from woof.core.kernels import load_module

    frame = load_module("ysu").get_function("ysu_column").local_size_bytes
    assert frame <= DEFAULT_STACK_BYTES, (
        f"ysu_column's per-thread frame is {frame} B, over the "
        f"{DEFAULT_STACK_BYTES} B default stack.  CUDA reserves "
        f"(frame - stack) x SMs x threads/SM of device memory at launch "
        f"and never returns it; on a 70 SM x 1,536 card that is "
        f"{(frame - DEFAULT_STACK_BYTES) * 70 * 1536 / 1024 / 1024:.1f} "
        f"MiB, charged to every run that selects bl_pbl_physics = 1 -- "
        f"which is the wizard's default.")


def _demo_columns(nz, ny, nx, seed=3):
    import cupy as cp
    from woof.core.state import DTYPE

    rng = np.random.default_rng(seed)

    def f3(lo, hi, n=nz):
        return cp.asarray(rng.uniform(lo, hi, (n, ny, nx)).astype(DTYPE))

    def f2(lo, hi):
        return cp.asarray(rng.uniform(lo, hi, (ny, nx)).astype(DTYPE))

    psf = 1.0e5 + rng.uniform(-2e3, 2e3, (ny, nx))
    pif = np.empty((nz + 1, ny, nx), dtype=DTYPE)
    for k in range(nz + 1):
        pif[k] = psf * (1.0 - 0.92 * k / nz)
    pr = np.empty((nz, ny, nx), dtype=DTYPE)
    for k in range(nz):
        pr[k] = 0.5 * (pif[k] + pif[k + 1])
    th = np.empty((nz, ny, nx), dtype=DTYPE)
    base = 290.0 + rng.uniform(-5, 5, (ny, nx))
    for k in range(nz):
        th[k] = base + 3.2 * k / nz * 10.0
    col = dict(u=f3(-12, 12), v=f3(-12, 12), theta=cp.asarray(th),
               qv=f3(1e-4, 1.2e-2), qc=f3(0, 3e-5), qi=f3(0, 2e-5),
               p=cp.asarray(pr), p_interface=cp.asarray(pif),
               exner=cp.asarray((pr / 1.0e5) ** 0.2857).astype(DTYPE),
               dz=f3(20, 250), rthraten=f3(-8e-4, 2e-4))
    surf = dict(psfc=cp.asarray(psf.astype(DTYPE)), znt=f2(0.01, 0.9),
                ust=f2(0.05, 0.85), hfx=f2(-30, 250), qfx=f2(-1e-5, 2e-4),
                wspd=f2(0.7, 15), br=f2(-2.0, 0.6), psim=f2(0.4, 4.0),
                psih=f2(0.4, 4.0), xland=f2(1.0, 2.0), u10=f2(-8, 8),
                v10=f2(-8, 8))
    return col, surf


def _bits(out):
    import cupy as cp

    return {k: cp.asnumpy(v).copy() for k, v in out.items()}


def _assert_bitwise(a, b, why):
    for name in a:
        assert a[name].dtype == b[name].dtype
        assert np.array_equal(a[name].view(np.uint32) if
                              a[name].dtype != np.int32 else a[name],
                              b[name].view(np.uint32) if
                              b[name].dtype != np.int32 else b[name]), (
            f"{name} moved: {why}")


@pytest.mark.gpu
def test_the_workspace_is_free_of_residue():
    """No column array may be read before it is written."""
    import cupy as cp
    from woof.core import ysu as Y

    col, surf = _demo_columns(37, 8, 8)
    real_empty = cp.empty

    def poisoned(value):
        def _empty(shape, dtype=float, *a, **kw):
            arr = real_empty(shape, dtype=dtype, *a, **kw)
            arr.fill(dtype(value) if dtype != cp.int32 else 0)
            return arr
        return _empty

    seen = {}
    for value in (-7.0e30, 3.5e30):
        cp.empty = poisoned(value)
        try:
            seen[value] = _bits(Y.launch_ysu(**col, **surf, dt=30.0))
        finally:
            cp.empty = real_empty
    a, b = (seen[v] for v in (-7.0e30, 3.5e30))
    _assert_bitwise(a, b,
                    "the workspace carries residue between uses, so some "
                    "column array is read before it is written")


@pytest.mark.gpu
def test_tiling_does_not_change_the_answer():
    """A tile boundary must not drop or double-count a column."""
    from woof.core import ysu as Y

    col, surf = _demo_columns(41, 12, 12)
    whole = _bits(Y.launch_ysu(**col, **surf, dt=30.0))

    original = Y.ysu_tile_columns
    try:                        # force many tiles, incl. a partial last one
        Y.ysu_tile_columns = lambda fn, ncol: Y.YSU_BLOCK
        tiled = _bits(Y.launch_ysu(**col, **surf, dt=30.0))
    finally:
        Y.ysu_tile_columns = original
    _assert_bitwise(whole, tiled,
                    "tiling changed the result, so a tile boundary drops "
                    "or double-counts columns")
