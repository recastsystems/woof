"""The fused tendency coupling writes exactly what the array expressions wrote.

``couple_ysu_tendencies``, ``couple_column_tendencies`` and
``_couple_momentum_to_faces`` take one launch each on CuPy arrays
(``woof.core.tendency_coupling``) and keep their CuPy expressions for any
other carrier.  This runs both paths of each on the same inputs, for every
combination of specified, nested, open-x, open-y and map-factor domains, and
compares every returned array as uint32.  The rates and map factors carry
signed zeros, below-normal-range values, infinities and NaN, and some map
factors are zero, so a zeroed ring element divided by its map factor is
checked too.
"""
from __future__ import annotations

import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu

_FIELDS = ("ru", "rv", "rtheta", "rqv", "rqc", "rqr", "rqi", "rqs")
_FLAGS = list(itertools.product([False, True], repeat=5))


def _salted(rng, shape, scale):
    f32 = np.float32
    a = (rng.standard_normal(shape) * scale).astype(f32)
    flat = a.reshape(-1)
    special = np.array([0.0, -0.0, 1.0e-40, -1.0e-40, np.inf, -np.inf,
                        np.nan], dtype=f32)
    at = rng.choice(flat.size, size=4 * special.size, replace=False)
    flat[at] = np.tile(special, 4)
    return a


def _state(cp, rng, nz, ny, nx, has_msf):
    f32 = np.float32
    mub2d = cp.asarray(rng.uniform(7.0e4, 1.0e5, (ny, nx)).astype(f32))
    mup = cp.asarray(_salted(rng, (ny, nx), 300.0))

    def factor(shape):
        m = rng.uniform(0.9, 1.1, shape).astype(f32)
        m.reshape(-1)[rng.choice(m.size, size=3, replace=False)] = [
            0.0, -0.0, np.nan]
        return cp.asarray(m)

    return SimpleNamespace(
        p=cp.empty((nz, ny, nx), dtype=cp.float32),
        c1h=cp.asarray(rng.uniform(0.0, 1.0, nz).astype(f32)),
        c2h=cp.asarray(rng.uniform(0.0, 2.0e4, nz).astype(f32)),
        total_mu=lambda: mub2d + mup,
        has_msf=has_msf,
        msft=factor((ny, nx)), msfu=factor((ny, nx + 1)),
        msfv=factor((ny + 1, nx)))


def _bits(cp, tendencies):
    out = {}
    for name in _FIELDS:
        value = getattr(tendencies, name)
        out[name] = None if value is None else cp.asnumpy(
            cp.ascontiguousarray(value)).view(np.uint32)
    return out


def _assert_same(got, want, label):
    for name in _FIELDS:
        if want[name] is None:
            assert got[name] is None, f"{label} {name}: expected None"
            continue
        assert got[name].shape == want[name].shape, (label, name)
        differing = int(np.count_nonzero(got[name] != want[name]))
        assert differing == 0, f"{label} {name}: {differing} words differ"


def _both(monkeypatch, run):
    from woof.core import tendency_coupling

    launches = []
    real_mass, real_faces = (tendency_coupling.couple_mass,
                             tendency_coupling.couple_faces)
    monkeypatch.setattr(tendency_coupling, "couple_mass",
                        lambda *a, **k: (launches.append("mass"),
                                         real_mass(*a, **k))[1])
    monkeypatch.setattr(tendency_coupling, "couple_faces",
                        lambda *a, **k: (launches.append("faces"),
                                         real_faces(*a, **k))[1])
    fused = run()
    monkeypatch.setattr(tendency_coupling, "ready", lambda *a, **k: False)
    reference = run()
    monkeypatch.undo()
    return fused, reference, launches


@pytest.mark.gpu
@requires_gpu
def test_reused_pbl_outputs_overwrite_composed_values():
    """An old radiation/cumulus sum must not leak into fresh coupled rates."""
    import cupy as cp
    from woof.core import physics

    rng = np.random.default_rng(20261101)
    state = _state(cp, rng, 5, 9, 12, True)
    cfg = SimpleNamespace(specified=False, nested=False, open_x=False, open_y=False)
    names = ("du", "dv", "dtheta", "dqv", "dqc", "dqi")
    rates = {name: cp.asarray(_salted(rng, state.p.shape, 1.0e-3)) for name in names}
    old = physics.couple_ysu_tendencies(state, cfg, rates)
    pointers = {name: getattr(old, name).data.ptr for name in _FIELDS
                if getattr(old, name) is not None}
    for value in vars(old).values():
        if isinstance(value, cp.ndarray):
            value.fill(cp.float32(12345.0))
    fresh = {name: cp.asarray(_salted(rng, state.p.shape, 2.0e-3)) for name in names}
    want = _bits(cp, physics.couple_ysu_tendencies(state, cfg, fresh))
    got = physics.couple_ysu_tendencies(state, cfg, fresh, out=old)
    _assert_same(_bits(cp, got), want, "reused coupled rates")
    assert all(getattr(got, name).data.ptr == ptr for name, ptr in pointers.items())
    # A model without prognostic ice must drop the old ice slot.
    fresh.pop("dqi")
    got = physics.couple_ysu_tendencies(state, cfg, fresh, out=got)
    assert got.rqi is None


@pytest.mark.parametrize("cadence,relocated,is_hip,reused", [
    (0.0, False, True, True), (1.0, False, True, False), (0.0, True, True, False),
    # CUDA never reuses the held outputs: its path stays the release path.
    (0.0, False, False, False),
])
def test_pbl_output_reuse_respects_cadence_and_state(monkeypatch, cadence,
                                                   relocated, is_hip, reused):
    from woof.config import RunConfig
    from woof.core import physics

    # The reuse is HIP-only; name the runtime instead of inheriting the host's.
    monkeypatch.setattr(physics.cp.cuda.runtime, "is_hip", is_hip, raising=False)

    state = SimpleNamespace(qi=None)
    previous = object()
    driver = SimpleNamespace(state=state, terrain_drag=None, gf_rthblten=None,
                             fields={}, pbl_tendencies=previous)
    cfg = RunConfig(nx=12, ny=10, nz=8, dx=3000.0, dy=3000.0,
                    ztop=18000.0, dt=2.0, run_seconds=4.0, bl_pbl_physics=1,
                    ra_physics=1, bldt=cadence)
    seen = []
    monkeypatch.setattr(physics, "couple_ysu_tendencies",
                        lambda *args, out=None: seen.append(out))
    active = SimpleNamespace(qi=None) if relocated else state
    physics.PhysicsDriver._couple_pbl_slot(driver, cfg, {}, state=active)
    assert seen == [previous if reused else None]


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("flags", _FLAGS)
def test_fused_pbl_coupling_is_bitwise_the_expressions(monkeypatch, flags):
    cp = pytest.importorskip("cupy")
    from woof.core import physics

    specified, nested, open_x, open_y, has_msf = flags
    rng = np.random.default_rng(20260930 + sum(b << i for i, b in
                                               enumerate(flags)))
    nz, ny, nx = 5, 9, 12
    state = _state(cp, rng, nz, ny, nx, has_msf)
    cfg = SimpleNamespace(specified=specified, nested=nested, open_x=open_x,
                          open_y=open_y)
    for with_qi in (True, False):
        rates = {name: cp.asarray(_salted(rng, (nz, ny, nx), 1.0e-3))
                 for name in ("du", "dv", "dtheta", "dqv", "dqc", "dqi")}
        if not with_qi:
            rates.pop("dqi")
        fused, reference, launches = _both(
            monkeypatch,
            lambda: _bits(cp, physics.couple_ysu_tendencies(state, cfg,
                                                            rates)))
        assert launches == ["mass", "faces"]
        _assert_same(fused, reference, f"ysu {flags} qi={with_qi}")


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("flags", _FLAGS)
def test_fused_column_coupling_is_bitwise_the_expressions(monkeypatch,
                                                          flags):
    cp = pytest.importorskip("cupy")
    from woof.core import physics

    specified, nested, open_x, open_y, has_msf = flags
    rng = np.random.default_rng(20261001 + sum(b << i for i, b in
                                               enumerate(flags)))
    nz, ny, nx = 4, 7, 10
    state = _state(cp, rng, nz, ny, nx, has_msf)
    cfg = SimpleNamespace(specified=specified, nested=nested, open_x=open_x,
                          open_y=open_y)
    names = ("rtheta", "rqv", "rqc", "rqr", "rqi", "rqs")
    for keep in ((1, 1, 1, 0, 0, 0), (1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0),
                 (0, 1, 0, 1, 0, 1), (1, 0, 1, 0, 1, 0)):
        for momentum in (False, True):
            kwargs = {name: cp.asarray(_salted(rng, (nz, ny, nx), 1.0e-3))
                      for name, k in zip(names, keep) if k}
            if momentum:
                kwargs["ru"] = cp.asarray(_salted(rng, (nz, ny, nx), 1.0e-3))
                kwargs["rv"] = cp.asarray(_salted(rng, (nz, ny, nx), 1.0e-3))
            fused, reference, launches = _both(
                monkeypatch,
                lambda: _bits(cp, physics.couple_column_tendencies(
                    state, cfg, **kwargs)))
            assert launches == (["mass", "faces"] if momentum else ["mass"])
            _assert_same(fused, reference,
                         f"column {flags} keep={keep} momentum={momentum}")
