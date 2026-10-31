"""The duplicated staggered face of a periodic domain equals face 0.

An x-staggered array on a doubly periodic domain carries ``nx + 1`` slots
for ``nx`` distinct faces: the last one IS the first one, wrapped.  Every
consumer that destaggers to cell centres reads
``0.5*(u[..., nx-1] + u[..., nx])`` for the last mass column, so a
duplicate slot that stops being written does not crash and does not go
non-finite -- it quietly makes the domain's last column read a wind from
whenever the slot was last touched.  A 4-GPU fork carried exactly that
defect, and the question of whether mainline shares it (task #143) was
answered by a 1,200-measurement probe on node 3: mainline is CLEAN, the
identity holds at exactly 0.0 in FP32 while the slot itself travelled up
to 2.03 m/s.

That was a probe against an installed 1.8.7 wheel and it left with the
run.  This is the mainline pin, in the shape the probe used:

* the face arrays are found BY STRUCTURE -- every ``cupy.ndarray`` on the
  state whose trailing extents are ``(ny, nx+1)`` or ``(ny+1, nx)`` -- so
  an array a later lane adds to ``DomainState`` is covered the day it
  lands, rather than the day someone remembers to extend a hand-list;
* the identity is asserted EXACTLY, not to a tolerance, because the
  duplicate slot is computed from the same wrapped mass-point indices as
  face 0 and any difference at all is a different code path;
* a counterfactual says how far the slot moved, so a frozen field cannot
  pass as a held identity;
* and a poison control proves the assert can fire, run after the
  integration so the detector is exercised on evolved arrays.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]

_NX = _NY = 64
_NZ = 30
_STEPS = 300
_DT = 2.0
_SURFACE_THETA = 300.0
_LAPSE = 0.003
_SURFACE_SKIN = 305.0


def _face_arrays(obj, nx: int, ny: int):
    """Every array on ``obj`` carrying a duplicated periodic face.

    Structure, not a name list: ``(..., ny, nx+1)`` is x-staggered and
    ``(..., ny+1, nx)`` is y-staggered.  On the shipped ``DomainState``
    this finds u/u0/u_pp/ru_t/msfu and the v family.
    """
    import cupy as cp

    xface, yface = {}, {}
    for name in sorted(dir(obj)):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:                                   # noqa: BLE001
            continue
        if not isinstance(value, cp.ndarray) or value.ndim < 2:
            continue
        if value.shape[-1] == nx + 1 and value.shape[-2] == ny:
            xface[name] = value
        elif value.shape[-1] == nx and value.shape[-2] == ny + 1:
            yface[name] = value
    return xface, yface


def _alias_diffs(obj, nx: int, ny: int) -> dict[str, float]:
    """``max |duplicate face - face 0|`` for every face array."""
    import cupy as cp

    xface, yface = _face_arrays(obj, nx, ny)
    out = {f"x:{name}": float(cp.abs(a[..., -1] - a[..., 0]).max())
           for name, a in xface.items()}
    out.update({f"y:{name}": float(cp.abs(a[..., -1, :] - a[..., 0, :]).max())
                for name, a in yface.items()})
    return out


class _Shim:
    """Attribute bag, so the poison control runs the identical sweep."""


def _poisoned(state, nx: int, ny: int, offset: float = 0.5):
    """Every face array copied and its duplicate slot corrupted.

    ``cupy.copy`` first: the live state is never touched, so the poison
    cannot leak into the assertions that follow it.
    """
    import cupy as cp

    shim = _Shim()
    xface, yface = _face_arrays(state, nx, ny)
    for name, a in xface.items():
        bad = cp.copy(a)
        bad[..., -1] += a.dtype.type(offset)
        setattr(shim, name, bad)
    for name, a in yface.items():
        bad = cp.copy(a)
        bad[..., -1, :] += a.dtype.type(offset)
        setattr(shim, name, bad)
    return shim


def _periodic_state():
    """A small doubly periodic full-physics case, faces equal at init."""
    import cupy as cp

    from woof.config import RunConfig, validate_run_config
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.physics import (initialize_physics,
                                    physics_driver_required)
    from woof.core.state import init_at_rest

    # open_x = False is periodic (woof/config.py), and the default.
    cfg = validate_run_config(RunConfig(
        nx=_NX, ny=_NY, nz=_NZ, dx=500.0, dy=500.0, ztop=2000.0,
        dt=_DT, run_seconds=_DT * _STEPS, time_step_sound=4,
        moist=True, mp_physics=1, bl_pbl_physics=1, sf_sfclay_physics=91,
        sf_surface_physics=0, km_opt=4, c_s=0.25,
        bldt=0.0, radt=0.0, cu_physics=0))
    assert cfg.open_x is False and cfg.open_y is False, "must be periodic"

    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(
        coord, lambda z: _SURFACE_THETA + _LAPSE * np.asarray(z, float),
        p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_at_rest(cfg, coord, base)

    rng = np.random.default_rng(20260810)
    u = np.full(tuple(state.u.shape), 6.0) + 0.05 * rng.standard_normal(
        tuple(state.u.shape))
    u[:, :, -1] = u[:, :, 0]                  # identity true at init
    state.u[...] = cp.asarray(u, dtype=state.u.dtype)
    v = np.full(tuple(state.v.shape), 3.0) + 0.05 * rng.standard_normal(
        tuple(state.v.shape))
    v[:, -1, :] = v[:, 0, :]
    state.v[...] = cp.asarray(v, dtype=state.v.dtype)
    thp = np.zeros(tuple(state.thp.shape))
    thp[:4] = 0.1 * rng.standard_normal(thp[:4].shape)
    state.thp[...] = cp.asarray(thp, dtype=state.thp.dtype)
    state.qv[...] = cp.asarray(
        np.full(tuple(state.qv.shape), 0.006), dtype=state.qv.dtype)

    if physics_driver_required(cfg):
        initialize_physics(state, cfg, landmask=1.0, tsk=_SURFACE_SKIN,
                           glw=0.0)
    return state, cfg


def test_the_periodic_duplicate_face_is_written_every_step():
    """The identity, exactly, on every duplicated-face array in the state."""
    import cupy as cp

    from woof.core.dycore import run_steps, stability_report

    state, cfg = _periodic_state()
    nx, ny = cfg.nx, cfg.ny

    xface, yface = _face_arrays(state, nx, ny)
    # The set is discovered, but it is not allowed to be EMPTY or to lose
    # a member silently -- an empty sweep is the way a pin stops pinning.
    assert set(xface) >= {"u", "u0", "u_pp", "ru_t", "msfu"}, sorted(xface)
    assert set(yface) >= {"v", "v0", "v_pp", "rv_t", "msfv"}, sorted(yface)
    # The stored MF_VX_INV plane (paper/conus-suite-exact, 2.9.11) is a
    # y-face array with no x twin: WRF stores the inverse of msfv only.
    assert "msfv_inv" in yface and "msfu_inv" not in xface
    assert len(xface) == len(yface) - 1 >= 5

    # Control (a): true at init, before anything has run.
    assert all(diff == 0.0
               for diff in _alias_diffs(state, nx, ny).values())

    u_slot_init = cp.copy(state.u[:, :, -1])
    v_slot_init = cp.copy(state.v[:, -1, :])

    run_steps(state, cfg, _STEPS)
    cp.cuda.runtime.deviceSynchronize()

    health = {k: float(v) for k, v in stability_report(state, cfg).items()}
    assert health["nan"] == 0.0, health
    assert health["cfl"] < 1.0, health

    diffs = _alias_diffs(state, nx, ny)
    assert len(diffs) >= 10
    bad = {name: value for name, value in diffs.items() if value != 0.0}
    assert not bad, f"periodic face identity broken: {bad}"

    # The physical consequence a stale slot would inject, read directly:
    # the last mass column's cell-centre wind.
    u_as_read = 0.5 * (state.u[:, :, nx - 1] + state.u[:, :, nx])
    u_periodic = 0.5 * (state.u[:, :, nx - 1] + state.u[:, :, 0])
    v_as_read = 0.5 * (state.v[:, ny - 1, :] + state.v[:, ny, :])
    v_periodic = 0.5 * (state.v[:, ny - 1, :] + state.v[:, 0, :])
    assert float(cp.abs(u_as_read - u_periodic).max()) == 0.0
    assert float(cp.abs(v_as_read - v_periodic).max()) == 0.0

    # Control (c): the zero is a held identity, not a dead field.  The
    # duplicate slot itself has to have MOVED, or two frozen slots would
    # agree and this whole test would pass on a corpse.
    u_moved = float(cp.abs(state.u[:, :, -1] - u_slot_init).max())
    v_moved = float(cp.abs(state.v[:, -1, :] - v_slot_init).max())
    assert u_moved > 0.05, u_moved
    assert v_moved > 0.05, v_moved

    # Control (b): the same sweep on a deliberately corrupted copy, run
    # AFTER the integration so the detector is exercised on evolved
    # arrays rather than on zeros.
    poisoned = _alias_diffs(_poisoned(state, nx, ny, 0.5), nx, ny)
    assert set(poisoned) == set(diffs)
    for name, value in poisoned.items():
        assert value == pytest.approx(0.5, rel=1e-3), (name, value)
    # ... and the live state is untouched by the poison.
    assert all(v == 0.0 for v in _alias_diffs(state, nx, ny).values())
