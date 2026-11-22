"""The multi-card split of the donor guard and the reserve cap gives the one-card kernels' bytes.

On several cards a coarse column's factor is formed by the card that owns the
column and applied by every card that holds one of its edges
(``cuda_fullstep_lts_multi``).  These tests run the one-card kernel and the
factor-then-apply pair on the same data and require identical arrays.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

gpu = pytest.mark.skipif(os.environ.get("HEX_LTS_GPU_TESTS") != "1", reason="requires an explicit GPU job (HEX_LTS_GPU_TESTS=1)")


def _kernels():
    from woof.hex import cuda_fullstep_lts_multi as multi
    from woof.hex import cuda_fullstep_lts_scalars as scalars
    from woof.hex.cuda_backend import require_cuda
    from woof.hex.cuda_backend.runtime import KernelCache

    cache = KernelCache(capability=require_cuda(min_compute=(12, 0)))
    found = {name: cache.raw_kernel(name, scalars._SOURCE, module_key="hexcore.cuda_fullstep_lts_scalars")
             for name in scalars._KERNEL_NAMES}
    found.update({name: cache.raw_kernel(name, multi._MULTI_SOURCE, module_key="hexcore.cuda_fullstep_lts_multi")
                  for name in multi._MULTI_KERNELS})
    return found


def _launch(kernel, total, args):
    if total:
        kernel(((int(total) + 255) // 256,), (256,), args)


@gpu
def test_split_donor_guard_equals_the_one_card_guard():
    import cupy as cp

    k = _kernels()
    rng = np.random.default_rng(29)
    ntr, nlev, ncells, max_edges, nslots = 3, 7, 12, 3, 9
    # a closed strip: cell c has edges 2c-1, 2c, 2c+1 (mod), so neighbouring rows share edges
    nedges = 2 * ncells
    edges_on_cell = np.array([[(2 * c - 1) % nedges, 2 * c, (2 * c + 1) % nedges] for c in range(ncells)], np.int32)
    acoustic_sign = rng.choice([-1.0, 1.0], size=(ncells, max_edges)).astype(np.float32)
    counts = np.full(ncells, max_edges, np.int32)
    inv_area = rng.uniform(0.3, 0.9, ncells).astype(np.float32)
    rdzw = rng.uniform(0.5, 1.5, nlev).astype(np.float32)
    source_old = rng.uniform(0.0, 1.0, (ntr, nlev, ncells)).astype(np.float32)
    rho_old = rng.uniform(0.8, 1.2, (nlev, ncells)).astype(np.float32)
    ww = (0.6 * rng.normal(size=(nlev + 1, ncells))).astype(np.float32)
    upwind = rng.normal(size=(ntr, nlev, nedges)).astype(np.float32)
    residual = rng.normal(size=(ntr, nlev, nedges)).astype(np.float32)
    mass = rng.uniform(0.5, 1.5, (ntr, nlev, ncells)).astype(np.float32)
    vres = rng.normal(size=(ntr, nlev + 1, ncells)).astype(np.float32)
    rows = np.array([1, 4, 5, 9], np.int32)               # guard rows, two of them neighbours
    width = 2
    slot = rng.integers(0, nslots, size=(rows.size, width)).astype(np.int32)
    coef = rng.choice([-0.7, 0.4], size=(rows.size, width))
    coef[3, 1] = 0.0
    register = rng.uniform(0.0, 4.0, (ntr, nlev, nslots))
    air = rng.uniform(-1.0, 2.0, (nlev, nslots))
    packet = np.zeros(nedges, np.uint8)
    packet[[2, 8, 11]] = 1
    i32 = np.int32
    common = dict(counts=cp.asarray(counts), eoc=cp.asarray(edges_on_cell.reshape(-1)),
                  sign=cp.asarray(acoustic_sign.reshape(-1)), inv=cp.asarray(inv_area), packet=cp.asarray(packet),
                  src=cp.asarray(source_old), rho=cp.asarray(rho_old), ww=cp.asarray(ww), rdzw=cp.asarray(rdzw),
                  rows=cp.asarray(rows), slot=cp.asarray(slot.reshape(-1)), coef=cp.asarray(coef.reshape(-1)),
                  reg=cp.asarray(register), air=cp.asarray(air))
    one = {name: cp.asarray(v) for name, v in dict(up=upwind, res=residual, mass=mass, vres=vres).items()}
    stats1 = cp.zeros(4, dtype=cp.uint64)
    _launch(k["lts_donor_guard"], ntr * rows.size, (
        i32(ntr), i32(nlev), i32(rows.size), i32(width), i32(ncells), i32(nedges), i32(nslots), i32(max_edges),
        np.float32(1.3), common["rows"], common["slot"], common["coef"], common["reg"], common["air"],
        common["counts"], common["eoc"], common["sign"], common["inv"], common["packet"], common["src"],
        common["rho"], common["ww"], common["rdzw"], one["up"], one["res"], one["mass"], one["vres"],
        stats1, cp.zeros(3 * ntr, dtype=cp.float64)))
    two = {name: cp.asarray(v) for name, v in dict(up=upwind, res=residual, mass=mass, vres=vres).items()}
    keep = cp.ones((ntr, nlev, ncells), dtype=cp.float32)
    stats2 = cp.zeros(4, dtype=cp.uint64)
    _launch(k["lts_guard_keep"], ntr * rows.size, (
        i32(ntr), i32(nlev), i32(rows.size), i32(width), i32(ncells), i32(nedges), i32(nslots), i32(max_edges),
        np.float32(1.3), common["rows"], common["slot"], common["coef"], common["reg"], common["air"],
        common["counts"], common["eoc"], common["sign"], common["inv"], common["packet"], common["src"],
        common["rho"], common["ww"], common["rdzw"], two["up"], two["mass"], two["vres"], keep,
        stats2, cp.zeros(3 * ntr, dtype=cp.float64)))
    _launch(k["lts_guard_apply"], ntr * nlev * rows.size, (
        i32(ntr), i32(nlev), i32(rows.size), i32(ncells), i32(nedges), i32(max_edges), common["rows"],
        common["counts"], common["eoc"], common["sign"], common["packet"], keep, two["up"], two["res"]))
    assert int(cp.asnumpy(stats1)[0]) > 0
    np.testing.assert_array_equal(cp.asnumpy(stats1), cp.asnumpy(stats2))
    for name in one:
        np.testing.assert_array_equal(cp.asnumpy(one[name]), cp.asnumpy(two[name]), err_msg=name)


@gpu
def test_split_reserve_cap_equals_the_one_card_cap():
    import cupy as cp

    k = _kernels()
    rng = np.random.default_rng(31)
    ntr, nlev, rows, width, nedges, nslots, ncells = 3, 5, 6, 3, 40, 18, 30
    slot = rng.permutation(nslots).reshape(rows, width).astype(np.int32)
    coef = rng.choice([-0.5, 0.25], size=(rows, width))
    coef[1, 2] = coef[4, 0] = 0.0
    edge_of_slot = rng.permutation(nedges)[:nslots].astype(np.int32)
    cells = rng.permutation(ncells)[:rows].astype(np.int32)
    upwind = rng.normal(size=(ntr, nlev, nedges)).astype(np.float32)
    residual = (0.4 * rng.normal(size=(ntr, nlev, nedges))).astype(np.float32)
    reserve = rng.uniform(-0.2, 2.0, size=(ntr, nlev, rows))
    content = np.abs(reserve) * rng.uniform(0.5, 1.5, size=reserve.shape)
    air_start = rng.uniform(0.2, 1.0, size=(nlev, rows))
    reg_mass = rng.uniform(-0.5, 1.5, size=(nlev, nslots))
    margin = 1.0 - 2.0 ** -20
    i32 = np.int32
    table = (cp.asarray(slot.reshape(-1)), cp.asarray(coef.reshape(-1)), cp.asarray(edge_of_slot))
    up1, res1 = cp.asarray(upwind), cp.asarray(residual)
    hold1 = cp.zeros((ntr, nlev, nslots), dtype=cp.float32)
    stats1 = cp.zeros(4, dtype=cp.uint64)
    _launch(k["lts_reserve_cap"], ntr * nlev * rows, (
        i32(ntr), i32(nlev), i32(rows), i32(width), i32(nedges), i32(nslots), *table, i32(1), np.float64(margin),
        cp.asarray(reserve), cp.asarray(content), cp.asarray(air_start), cp.asarray(reg_mass),
        up1, res1, hold1, stats1, cp.zeros(3 * ntr, dtype=cp.float64)))
    up2, res2 = cp.asarray(upwind), cp.asarray(residual)
    hold2 = cp.zeros((ntr, nlev, nslots), dtype=cp.float32)
    low = cp.ones((ntr, nlev, ncells), dtype=cp.float64)
    high = cp.ones((ntr, nlev, ncells), dtype=cp.float64)
    stats2 = cp.zeros(4, dtype=cp.uint64)
    _launch(k["lts_cap_keep"], ntr * nlev * rows, (
        i32(ntr), i32(nlev), i32(rows), i32(width), i32(nedges), i32(nslots), i32(ncells), cp.asarray(cells),
        *table, i32(1), np.float64(margin), cp.asarray(reserve), cp.asarray(content), cp.asarray(air_start),
        cp.asarray(reg_mass), up2, res2, low, high, stats2, cp.zeros(3 * ntr, dtype=cp.float64)))
    _launch(k["lts_cap_apply"], ntr * nlev * rows, (
        i32(ntr), i32(nlev), i32(rows), i32(width), i32(nedges), i32(ncells), cp.asarray(cells), *table,
        low, high, up2, res2))
    _launch(k["lts_hold"], ntr * nlev * rows, (
        i32(ntr), i32(nlev), i32(rows), i32(width), i32(nedges), i32(nslots), *table, up2, hold2))
    assert int(cp.asnumpy(stats1)[2:].sum()) > 0
    np.testing.assert_array_equal(cp.asnumpy(stats1), cp.asnumpy(stats2))
    np.testing.assert_array_equal(cp.asnumpy(up1), cp.asnumpy(up2))
    np.testing.assert_array_equal(cp.asnumpy(res1), cp.asnumpy(res2))
    np.testing.assert_array_equal(cp.asnumpy(hold1), cp.asnumpy(hold2))
