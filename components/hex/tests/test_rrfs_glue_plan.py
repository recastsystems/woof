"""RRFS resident glue copy plans (woof.hex.mpasphys.rrfs.glue_plan), used by the fused MYNN-EDMF and RUC slot glue.

Breakage each check prevents:
- a plan op that writes a different word than the CuPy assignment it replaces (transposed core copies, broadcasts,
  fills, NaN payloads) would change the PBL or RUC inputs and outputs of an hrrrv5 forecast;
- a dtype conversion inside the glue would change words silently, so the plan refuses it;
- two ops of one launch that touch the same word would race; check_disjoint must refuse that plan.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.cuda


def _cupy():
    cp = pytest.importorskip('cupy')
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:  # noqa: BLE001 - no device on this host
        pytest.skip('no CUDA device')
    return cp


def test_plan_writes_the_assignment_words():
    cp = _cupy()
    from woof.hex.mpasphys.rrfs.glue_plan import CopyPlan
    rng = np.random.default_rng(1)
    whole = cp.empty((40, 7, 1), dtype=cp.float32, order='F')
    core = cp.asarray(rng.standard_normal((7, 40)).astype(np.float32))
    a = cp.asarray(rng.standard_normal((5, 30)).astype(np.float32))
    b = cp.zeros((30, 5), cp.float32)
    flags = cp.zeros((30, 1), cp.int32, order='F')
    z = cp.asarray(rng.standard_normal((3, 30)).astype(np.float32))
    zt = cp.empty_like(z)
    nan = cp.asarray(np.array([0x7fc01234, 0xffc00001, 0x00000001, 0x80000000], dtype=np.uint32).view(np.float32))
    nant = cp.empty_like(nan)
    plan = CopyPlan('test')
    plan.copy(whole[:, :, 0].T, core)
    plan.copy(b, a.T)
    plan.fill(flags[:, 0:1], cp.int32(1))
    plan.copy(zt, cp.broadcast_to(z[:, :1], z.shape))
    plan.copy(nant, nan)
    plan.build()
    assert plan.check_disjoint()
    plan()
    assert cp.array_equal(whole[:, :, 0].T, core)
    assert cp.array_equal(b, a.T)
    assert bool((flags == 1).all())
    assert cp.array_equal(zt, cp.broadcast_to(z[:, :1], z.shape))
    assert cp.array_equal(nant.view(cp.uint32), nan.view(cp.uint32))


def test_plan_refuses_conversion_and_races():
    cp = _cupy()
    from woof.hex.mpasphys.rrfs.glue_plan import CopyPlan
    with pytest.raises(TypeError):
        CopyPlan('convert').copy(cp.zeros(4, cp.float32), cp.zeros(4, cp.int32))
    a = cp.zeros((5, 30), cp.float32)
    b = cp.zeros((30, 5), cp.float32)
    race = CopyPlan('race')
    race.copy(b[:, :2], a.T[:, :2])
    race.copy(b[:, 1:3], a.T[:, 1:3])
    race.build()
    with pytest.raises(RuntimeError):
        race.check_disjoint()
