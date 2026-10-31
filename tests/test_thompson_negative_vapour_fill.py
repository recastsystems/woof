"""mp_gt_driver's negative-vapour fill, word for word against WRF's own loop.

WRF v4.6.1 ``phys/module_mp_thompson.F:1350-1445``: in the driver's per-column
k loop, after ``qv(i,k,j) = qv1d(k)``, a negative ``qv1d(k)`` becomes
``MAX(1.E-7, 0.5*(qv(i,k-1,j) + qv(i,k+1,j)))`` for ``kts+1 < k < kte-2`` and
``1.E-7`` elsewhere.  The breakage this prevents: without the fill, a dry level
the advection took a few 1e-13 below zero stayed negative in WOOF while the
strict wrf.exe 4.6.1 set it to about 1e-3, and the strict UAE 9 km pair parted
at step 2.
"""

from __future__ import annotations

import numpy as np

from conftest import requires_gpu


def _wrf_loop(post: np.ndarray, entry: np.ndarray) -> np.ndarray:
    """The Fortran loop, literally: ``qv`` starts as the driver's INPUT array
    and is overwritten level by level; ``qv1d`` is the scheme's output."""
    nz = post.shape[0]
    qv = entry.copy()
    kts, kte = 1, nz
    for k in range(kts, kte + 1):
        i = k - 1
        qv[i] = post[i]
        if post[i] < np.float32(0.0):
            if kts + 1 < k < kte - 2:
                mean = np.float32(np.float32(0.5) * np.float32(qv[i - 1] + qv[i + 1]))
                qv[i] = max(np.float32(1.0e-7), mean)
            else:
                qv[i] = np.float32(1.0e-7)
    return qv


def _columns():
    rng = np.random.default_rng(461)
    nz, ncol = 44, 64
    entry = rng.uniform(1e-8, 2e-3, size=(nz, ncol)).astype(np.float32)
    post = entry.copy()                 # a column the scheme left untouched
    negative = np.float32(-3.6e-11)
    # every boundary of the fill window, adjacent negatives (the k-1 read
    # sees the filled word), a subnormal negative, -0.0 and a NaN.
    for col, levels in enumerate(([0], [1], [2], [40], [41], [43], [10, 11],
                                  [20, 21, 22], [nz - 4, nz - 3])):
        post[levels, col] = negative
    post[5, 20] = np.float32(-1e-40)
    post[6, 21] = np.float32(-0.0)
    post[7, 22] = np.float32(np.nan)
    post[8, 23] = np.float32(-5e-4)     # a mean below the 1.E-7 floor
    entry[9, 23] = np.float32(1e-8)
    post[9, 23] = np.float32(1e-8)
    post[7, 23] = np.float32(1e-8)
    return post, entry


def test_the_numpy_transcription_reads_filled_below_and_input_above():
    post, entry = _columns()
    want = _wrf_loop(post[:, 26:27], entry[:, 26:27])
    assert np.array_equal(want, entry[:, 26:27])          # nothing negative
    col = post[:, 7:8].copy(); col[[10, 11]] = np.float32(-1.0)
    got = _wrf_loop(col, entry[:, 7:8])
    first = np.float32(0.5) * np.float32(got[9, 0] + entry[11, 7])
    assert got[10, 0] == max(np.float32(1e-7), first)
    assert got[11, 0] == max(np.float32(1e-7),
                             np.float32(0.5) * np.float32(got[10, 0] + entry[12, 7]))


@requires_gpu
def test_the_kernel_is_wrfs_loop_word_for_word():
    import cupy as cp
    from woof.core.microphysics import thompson_negative_vapour_fill

    # The fill only ever meets a column mp_thompson left at its
    # no-microphysics exit (:2020), where qv1d IS the driver's input vapour,
    # negatives included: so WRF's loop runs on entry == post.
    post, _ = _columns()
    device = cp.asarray(post.reshape(post.shape[0], 8, 8))
    thompson_negative_vapour_fill(device)
    got = cp.asnumpy(device).reshape(post.shape)
    ref = np.stack([_wrf_loop(post[:, c:c + 1], post[:, c:c + 1])[:, 0]
                    for c in range(post.shape[1])], axis=1)
    assert np.array_equal(got.view(np.uint32), ref.view(np.uint32))
    # -0.0 and NaN are not < 0 and stay; the subnormal negative is filled;
    # a mean below the floor takes the floor.
    assert got[6, 21].view(np.uint32) == np.float32(-0.0).view(np.uint32)
    assert np.isnan(got[7, 22])
    assert got[5, 20] > 0
    assert got[8, 23] == np.float32(1e-7)


@requires_gpu
def test_the_wrf_44_column_generation_fills_after_its_kernel(monkeypatch):
    """WRF v4.4's mp_gt_driver carries the same loop (module_mp_thompson.F:
    1352-1362, the generation's pinned source c89bd2ec...).  The column
    kernel leaves a no-microphysics column's vapour as its input, so with
    the kernel stood in by a no-op the call must return WRF's filled words.
    The breakage this prevents: a ``thompson_version = "wrf_44"`` run kept
    a negative dry level that wrf.exe 4.4 sets to about 1e-3."""
    import cupy as cp
    from types import SimpleNamespace
    import woof.core.thompson_column as column
    import woof.core.kernels as kernels

    post, _ = _columns()
    nz = post.shape[0]
    monkeypatch.setattr(column, "table_identity", lambda cfg: None)
    monkeypatch.setattr(column, "table_root", lambda cfg: "stub")
    monkeypatch.setattr(column, "tables",
                        lambda root, device: {n: None for n in column.TABLE_ARGUMENTS})
    monkeypatch.setattr(kernels, "get_kernel",
                        lambda *a, **k: (lambda *x, **y: None))

    class State:
        def __init__(self):
            vol = lambda: cp.full((nz, 8, 8), 1.0e-3, cp.float32)
            self.p = cp.full((nz, 8, 8), 8.0e4, cp.float32)
            self.thp, self.w = vol() * 0, cp.zeros((nz + 1, 8, 8), cp.float32)
            self.thb, self.phb = cp.full(nz, 300.0, cp.float32), cp.zeros(nz + 1, cp.float32)
            self.php = cp.asarray(np.arange(nz + 1, dtype=np.float32)[:, None, None] * 1000 * np.ones((1, 8, 8), np.float32))
            for name in ("qc", "qi", "qr", "qs", "qg", "ni", "nr", "effc", "effi", "effs", "h_diabatic"):
                setattr(self, name, vol() * 0)
            self.qv = cp.asarray(post.reshape(nz, 8, 8))
            self.physics, self._s = SimpleNamespace(), {}

        def scratch(self, shape, slot):
            return self._s.setdefault(slot, cp.zeros(shape, cp.float32))

    state = State()
    cfg = SimpleNamespace(mp_physics=8, thompson_version="wrf_44",
                          no_mp_heating=1, mp_tend_lim=10.0)
    column.apply_generated(state, cfg, 10.0)
    got = cp.asnumpy(state.qv).reshape(post.shape)
    ref = np.stack([_wrf_loop(post[:, c:c + 1], post[:, c:c + 1])[:, 0]
                    for c in range(post.shape[1])], axis=1)
    assert np.array_equal(got.view(np.uint32), ref.view(np.uint32))
