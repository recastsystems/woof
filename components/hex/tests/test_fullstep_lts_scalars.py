"""The scalar interface law of active full-step LTS.

CPU: the law in plain arithmetic on the ring, through the production class
tables and event order -- one accepted packet per interface edge, spent in the
coarse cell's low-order mass, with the per-column reserve and donor guard.

GPU (HEX_LTS_GPU_TESTS=1 inside a mutex grant): the interface kernels against
the same arithmetic in numpy.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from woof.hex.fullstep_lts_classes import CLASS_MASK, SCRATCH_FLAG, macro_schedule, plan_macros, trajectory_plane
from test_fullstep_lts_classes import plan_for

gpu = pytest.mark.skipif(os.environ.get("HEX_LTS_GPU_TESTS") != "1", reason="requires an explicit GPU job (HEX_LTS_GPU_TESTS=1)")


def test_a_finer_class_ends_a_step_at_the_middle_and_end_of_every_coarser_step():
    # scalars are accepted at step ends only: stage two and three read the
    # consumer's middle, the committed block its end
    for consumer in (2, 4, 8, 16):
        for provider in (1, 2, 4, 8):
            if provider >= consumer:
                continue
            for count in (2, 4, 6):
                planes = 3 * count
                ends = [planes * provider * (i + 1) for i in range(consumer // provider)]
                found = [trajectory_plane(0, consumer, tick, planes) for tick in ends]
                assert planes // 2 in found and found[-1] == planes
    # two classes sharing a step: the provider has no middle state
    assert [trajectory_plane(0, 2, tick, 6) for tick in (12,)] == [6]
    assert [trajectory_plane(0, 2, tick, 12) for tick in (24,)] == [12]


def lts_fct(rates, steps, *, courant=0.2, packets=True, guard=True, cap=True, data="step", seed=5):
    """Flux-corrected transport on the ring with the interface law of the CUDA route.

    Unit density and area, uniform wind toward +index.  A class advances its
    owned cells and its scratch zone with an upwind low-order flux and a
    limited second-order residual.  On an edge the finer class owns, the
    finer class's flux (low order plus limited residual) is the packet: it is
    capped by the coarse cell's reserve, added to the register, and becomes
    the coarse class's low-order flux with no residual.  The coarse cell's own
    low-order outflow is then scaled to what the packets left.
    """

    rates = np.asarray(rates, dtype=np.int32)
    plan, coe = plan_for(rates, factor=1.5, min_rings=3)
    n = rates.size
    rng = np.random.default_rng(seed)
    if data == "step":
        truth = (np.arange(n) % 12 < 6).astype(np.float64)
    elif data == "constant":
        truth = np.ones(n)
    else:
        truth = rng.random(n)
    start_of = truth.copy()
    cls = plan.cell_class
    local = [truth[t.layout.cell_l2g].copy() for t in plan.classes]
    stands = np.zeros(len(plan.classes), dtype=int)
    register = np.zeros(plan.n_register_edges)
    reserve = [truth[t.layout.cell_l2g[t.reflux_cell_local]].copy() for t in plan.classes]
    initial = float(truth.sum())
    lowest = float(truth.min())
    events = {"cap": 0, "guard": 0}
    clock = 0
    for macro in plan_macros(steps, int(rates.max())):
        for kind, index, begin, length in macro_schedule(plan.rates, macro):
            tables = plan.classes[index]
            layout = tables.layout
            q = local[index]
            n_owned = layout.n_owned_cells
            g2l = layout.cell_g2l()
            e_l2g = layout.edge_l2g
            if kind == "sync":
                register[tables.coarse_slot] = 0.0
                truth[layout.cell_l2g[:n_owned]] = q[:n_owned]
                start_of[cls > index] = truth[cls > index]
                reserve[index] = q[tables.reflux_cell_local].copy()
                continue
            t0, c = clock + begin, courant * length
            assert stands[index] == t0
            halo = layout.cell_l2g[n_owned:]
            owner = tables.halo_cell_class & CLASS_MASK
            carried = (tables.halo_cell_class & SCRATCH_FLAG) != 0
            finer = owner > index
            tail = q[n_owned:]
            tail[finer] = start_of[halo[finer]]
            write = ~finer & (~carried | (stands[owner] == t0))
            tail[write] = truth[halo[write]]
            advance = np.ones(layout.n_local_cells, dtype=bool)
            advance[n_owned:] = carried
            start = q.copy()
            # every local edge with both cells local: low-order flux and residual, leaving cell e
            up, res = {}, {}
            for edge in e_l2g:
                a, b = g2l[coe[edge, 0]], g2l[coe[edge, 1]]
                if a < 0 or b < 0:
                    continue
                up[int(edge)] = c * start[a]
                res[int(edge)] = c * 0.5 * (1.0 - c) * (start[b] - start[a])
            packet_edges = {int(e_l2g[e]): int(s) for e, s in zip(tables.coarse_edge_local, tables.coarse_slot)}
            if packets:
                for edge, slot in packet_edges.items():
                    up[edge], res[edge] = register[slot], 0.0
                if guard:
                    for row, cell_local in enumerate(tables.reflux_cell_local):
                        cell = int(layout.cell_l2g[cell_local])
                        available = start[cell_local] + sum(
                            coef * register[slot] for slot, coef in zip(tables.reflux_slot[row], tables.reflux_coef[row]))
                        mine = [(edge, sign) for edge, sign in (((cell - 1) % n, -1.0), (cell, 1.0))
                                if edge not in packet_edges and sign * up[edge] > 0.0]
                        outflow = sum(sign * up[edge] for edge, sign in mine)
                        if outflow > available:
                            events["guard"] += 1
                            keep = max(available, 0.0) / outflow
                            for edge, _ in mine:
                                up[edge] *= keep
                                res[edge] *= keep
            fine_edges = {int(s): int(e_l2g[e]) for e, s in zip(tables.fine_edge_local, tables.fine_slot)}
            if fine_edges:
                parent = plan.classes[index - 1]
                for row in range(parent.reflux_cell_local.size):
                    mine = [(fine_edges[int(slot)], -1.0 if coef > 0 else 1.0)
                            for slot, coef in zip(parent.reflux_slot[row], parent.reflux_coef[row]) if coef != 0.0]
                    low = sum(max(sign * up[edge], 0.0) for edge, sign in mine)
                    high = sum(max(sign * res[edge], 0.0) for edge, sign in mine)
                    available = max(reserve[index - 1][row], 0.0) if cap else np.inf
                    keep_low = keep_high = 1.0
                    if low > available:
                        keep_low, keep_high = available / low, 0.0
                        events["cap"] += 1
                    elif low + high > available:
                        keep_high = (available - low) / high
                        events["cap"] += 1
                    for edge, sign in mine:
                        if sign * up[edge] > 0.0:
                            up[edge] *= keep_low
                        if sign * res[edge] > 0.0:
                            res[edge] *= keep_high
            # the limiter: low-order value and factors of every local cell with both edges local
            low_q = start.copy()
            r_in = np.ones(layout.n_local_cells)
            r_out = np.ones(layout.n_local_cells)
            for cell_local in range(layout.n_local_cells):
                cell = int(layout.cell_l2g[cell_local])
                left, right = (cell - 1) % n, cell
                if left not in up or right not in up:
                    continue
                low_q[cell_local] = start[cell_local] + up[left] - up[right]
                around = [start[cell_local], start[g2l[(cell - 1) % n]], start[g2l[(cell + 1) % n]]]
                incoming = max(res[left], 0.0) + max(-res[right], 0.0)
                outgoing = max(-res[left], 0.0) + max(res[right], 0.0)
                if incoming > 0.0:
                    r_in[cell_local] = min(1.0, max(0.0, (max(around) - low_q[cell_local]) / incoming))
                if outgoing > 0.0:
                    r_out[cell_local] = min(1.0, max(0.0, (low_q[cell_local] - min(around)) / outgoing))
            limited = {}
            for edge, value in res.items():
                a, b = g2l[coe[edge, 0]], g2l[coe[edge, 1]]
                limited[edge] = value * (min(r_out[a], r_in[b]) if value > 0.0 else min(r_in[a], r_out[b]))
            for cell_local in np.flatnonzero(advance):
                cell = int(layout.cell_l2g[cell_local])
                left, right = (cell - 1) % n, cell
                assert left in up and right in up, "an advanced cell read an edge that is not local"
                q[cell_local] = low_q[cell_local] + limited[left] - limited[right]
            if fine_edges:
                for row in range(parent.reflux_cell_local.size):
                    for slot, coef in zip(parent.reflux_slot[row], parent.reflux_coef[row]):
                        if coef == 0.0:
                            continue
                        edge = fine_edges[int(slot)]
                        packet = up[edge] + limited[edge]
                        register[slot] += packet
                        reserve[index - 1][row] -= packet if coef < 0 else -packet
            stands[index] = t0 + length
            lowest = min(lowest, float(q[:n_owned].min()))
        clock += macro
    total = float(sum(q[:t.layout.n_owned_cells].sum() for q, t in zip(local, plan.classes)))
    final = np.empty(n)
    for q, t in zip(local, plan.classes):
        final[t.layout.cell_l2g[:t.layout.n_owned_cells]] = q[:t.layout.n_owned_cells]
    return initial, total, final, lowest, register, events


RATE_MAPS = [[1] * 16 + [2] * 32,
             [2] * 20 + [1] * 12 + [2] * 16,
             [1] * 12 + [2] * 12 + [4] * 24 + [2] * 12]


@pytest.mark.parametrize("rates", RATE_MAPS)
@pytest.mark.parametrize("data", ["step", "random"])
@pytest.mark.parametrize("courant", [0.05, 0.2])
def test_packets_conserve_and_keep_the_tracer_non_negative(rates, data, courant):
    initial, final, q, lowest, register, _ = lts_fct(rates, 96, courant=courant, data=data)
    assert abs(final - initial) <= 256 * np.finfo(np.float64).eps * max(initial, 1.0)
    assert lowest >= -1e-13
    assert not register.any()
    # the instrument must see a planted defect: each class applying its own
    # flux to an interface edge loses the total
    initial, broken, *_ = lts_fct(rates, 96, courant=courant, data=data, packets=False)
    assert abs(broken - initial) > 1e-6 * max(initial, 1.0)


@pytest.mark.parametrize("rates", RATE_MAPS)
def test_a_constant_tracer_stays_constant_through_packets(rates):
    _, _, q, _, _, events = lts_fct(rates, 96, data="constant")
    np.testing.assert_allclose(q, 1.0, rtol=0, atol=256 * np.finfo(np.float64).eps)
    assert events == {"cap": 0, "guard": 0}


def test_the_active_path_refuses_transport_with_the_limiter_off_and_names_why():
    import inspect

    from woof.hex import cuda_fullstep_lts_active as active

    text = inspect.getsource(active.FullStepLtsActive.__init__)
    assert "config_scalar_advection=False" not in text       # the scalar refusal of the dry stage is gone
    assert "tracer mass would" in text and "limiter off" in text


# -- the interface kernels against numpy -------------------------------------------------

def _kernels():
    from woof.hex.cuda_backend import require_cuda
    from woof.hex.cuda_backend.runtime import KernelCache
    from woof.hex import cuda_fullstep_lts_scalars as scalars

    cache = KernelCache(capability=require_cuda(min_compute=(12, 0)))
    return {name: cache.raw_kernel(name, scalars._SOURCE, module_key="hexcore.cuda_fullstep_lts_scalars")
            for name in scalars._KERNEL_NAMES}


def _launch(kernel, total, args):
    kernel(((int(total) + 255) // 256,), (256,), args)


@gpu
def test_reserve_cap_capture_and_apply_match_numpy():
    import cupy as cp

    kernels = _kernels()
    rng = np.random.default_rng(11)
    ntr, nlev, rows, width, nedges, nslots = 3, 4, 7, 3, 40, 21
    slot = rng.permutation(nslots).reshape(rows, width).astype(np.int32)
    coef = rng.choice([-0.5, 0.25], size=(rows, width))
    coef[2, 2] = coef[5, 1] = 0.0                       # padding entries
    edge_of_slot = rng.permutation(nedges)[:nslots].astype(np.int32)
    upwind = rng.normal(size=(ntr, nlev, nedges)).astype(np.float32)
    residual = (0.3 * rng.normal(size=(ntr, nlev, nedges))).astype(np.float32)
    reserve = rng.uniform(-0.2, 2.0, size=(ntr, nlev, rows))
    margin = 1.0 - 2.0 ** -20
    up, res, hold = upwind.astype(np.float64), residual.astype(np.float64), np.zeros((ntr, nlev, nslots), np.float32)
    capped = [0, 0]
    for t in range(ntr):
        for k in range(nlev):
            for row in range(rows):
                real = [(int(slot[row, w]), 1.0 if coef[row, w] < 0 else -1.0) for w in range(width) if coef[row, w] != 0]
                available = max(reserve[t, k, row], 0.0) * margin
                low = sum(max(s * up[t, k, edge_of_slot[sl]], 0.0) for sl, s in real)
                high = sum(max(s * res[t, k, edge_of_slot[sl]], 0.0) for sl, s in real)
                keep_low = keep_high = 1.0
                if low > available:
                    keep_low, keep_high = available / low, 0.0
                    capped[0] += 1
                elif low + high > available:
                    keep_high = (available - low) / high
                    capped[1] += 1
                for sl, s in real:
                    e = edge_of_slot[sl]
                    if keep_low < 1.0 and s * up[t, k, e] > 0:
                        up[t, k, e] = np.float32(up[t, k, e] * keep_low)
                    if keep_high < 1.0 and s * res[t, k, e] > 0:
                        res[t, k, e] = np.float32(res[t, k, e] * keep_high)
                    hold[t, k, sl] = up[t, k, e]
    d_up, d_res = cp.asarray(upwind), cp.asarray(residual)
    d_hold = cp.zeros((ntr, nlev, nslots), dtype=cp.float32)
    d_reserve = cp.asarray(reserve)
    stats = cp.zeros(4, dtype=cp.uint64)
    amounts = cp.zeros(3 * ntr, dtype=cp.float64)
    i32 = np.int32
    table = (cp.asarray(slot.reshape(-1)), cp.asarray(coef.reshape(-1)), cp.asarray(edge_of_slot))
    # no dry mass has left any column: the allowance is the reserve alone
    quiet = (cp.asarray(np.maximum(reserve, 0.0)), cp.ones((nlev, rows), dtype=cp.float64),
             cp.zeros((nlev, nslots), dtype=cp.float64))
    _launch(kernels["lts_reserve_cap"], ntr * nlev * rows, (
        i32(ntr), i32(nlev), i32(rows), i32(width), i32(nedges), i32(nslots), *table,
        i32(1), np.float64(margin), d_reserve, *quiet, d_up, d_res, d_hold, stats, amounts))
    np.testing.assert_array_equal(cp.asnumpy(d_up), up.astype(np.float32))
    np.testing.assert_array_equal(cp.asnumpy(d_res), res.astype(np.float32))
    np.testing.assert_array_equal(cp.asnumpy(d_hold), hold)
    assert [int(v) for v in cp.asnumpy(stats)[2:]] == capped and min(capped) > 0
    # what may still leave a column through the interface fits its reserve
    for t in range(ntr):
        for k in range(nlev):
            for row in range(rows):
                real = [(int(slot[row, w]), 1.0 if coef[row, w] < 0 else -1.0) for w in range(width) if coef[row, w] != 0]
                worst = sum(max(s * up[t, k, edge_of_slot[sl]], 0.0) + max(s * res[t, k, edge_of_slot[sl]], 0.0)
                            for sl, s in real)
                assert worst <= max(reserve[t, k, row], 0.0) * (1 + 1e-6) + 1e-7
    # a column that has exported four times the dry mass it held may export four
    # times its start content: nothing of the same request is capped
    air_out = np.zeros((nlev, nslots))
    for row in range(rows):
        for w in range(width):
            if coef[row, w] != 0:
                air_out[:, slot[row, w]] = (4.0 / width + 1.0) * (1.0 if coef[row, w] < 0 else -1.0)
    roomy = np.full((ntr, nlev, rows), 4.0)
    free_up, free_res = cp.asarray(upwind), cp.asarray(residual)
    free_stats = cp.zeros(4, dtype=cp.uint64)
    request = np.abs(upwind).max() + np.abs(residual).max()
    _launch(kernels["lts_reserve_cap"], ntr * nlev * rows, (
        i32(ntr), i32(nlev), i32(rows), i32(width), i32(nedges), i32(nslots), *table,
        i32(1), np.float64(margin), cp.asarray(roomy), cp.asarray(roomy * request),
        cp.ones((nlev, rows), dtype=cp.float64), cp.asarray(air_out),
        free_up, free_res, cp.zeros((ntr, nlev, nslots), dtype=cp.float32), free_stats,
        cp.zeros(3 * ntr, dtype=cp.float64)))
    np.testing.assert_array_equal(cp.asnumpy(free_up), upwind)
    np.testing.assert_array_equal(cp.asnumpy(free_res), residual)
    assert not cp.asnumpy(free_stats).any()
    # capture: the packet joins the register and leaves the reserve
    d_register = cp.zeros((ntr, nlev, nslots), dtype=cp.float64)
    _launch(kernels["lts_packet_capture"], ntr * nlev * rows, (
        i32(ntr), i32(nlev), i32(rows), i32(width), i32(nedges), i32(nslots), *table,
        d_hold, d_res, d_register, d_reserve, amounts))
    register = np.zeros((ntr, nlev, nslots))
    left = reserve.copy()
    for row in range(rows):
        for w in range(width):
            if coef[row, w] == 0:
                continue
            sl = int(slot[row, w])
            packet = hold[:, :, sl].astype(np.float64) + res[:, :, edge_of_slot[sl]].astype(np.float32).astype(np.float64)
            register[:, :, sl] += packet
            left[:, :, row] -= packet if coef[row, w] < 0 else -packet
    np.testing.assert_array_equal(cp.asnumpy(d_register), register)
    np.testing.assert_allclose(cp.asnumpy(d_reserve), left, rtol=0, atol=1e-12)
    booked = cp.asnumpy(amounts).reshape(ntr, 3)
    np.testing.assert_allclose(booked[:, 0], np.abs(register).sum(axis=(1, 2)), rtol=1e-12)
    assert np.all(booked[:, 1] > 0.0) and not booked[:, 2].any()
    # apply: the coarse side's low-order flux is the register, its residual zero
    c_edge = rng.permutation(nedges)[:5].astype(np.int32)
    c_slot = rng.permutation(nslots)[:5].astype(np.int32)
    c_up = cp.asarray(rng.normal(size=(ntr, nlev, nedges)).astype(np.float32))
    c_res = cp.ones((ntr, nlev, nedges), dtype=cp.float32)
    _launch(kernels["lts_packet_apply"], ntr * nlev * 5, (
        i32(ntr), i32(nlev), i32(5), i32(nedges), i32(nslots), cp.asarray(c_edge), cp.asarray(c_slot),
        d_register, c_up, c_res))
    np.testing.assert_array_equal(cp.asnumpy(c_up)[:, :, c_edge], register[:, :, c_slot].astype(np.float32))
    assert not cp.asnumpy(c_res)[:, :, c_edge].any() and float(cp.asnumpy(c_res).sum()) == ntr * nlev * (nedges - 5)


@gpu
def test_donor_guard_removes_what_the_packets_took_beyond_the_upwind_export_and_moves_no_mass():
    import cupy as cp

    kernels = _kernels()
    rng = np.random.default_rng(13)
    ntr, nlev, ncells, nedges, max_edges, nslots = 2, 6, 5, 5, 2, 3
    # a ring of five cells; cell 2 is the interface column, its edge 2 (to cell 3) the packet edge
    edges_on_cell = np.array([[(c - 1) % 5, c] for c in range(5)], dtype=np.int32)
    acoustic_sign = np.tile(np.array([-1.0, 1.0], dtype=np.float32), (5, 1))   # +1: the edge leaves the cell
    inv_area = np.full(5, 0.5, dtype=np.float32)
    rdzw = rng.uniform(0.5, 1.5, nlev).astype(np.float32)
    source_old = rng.uniform(0.0, 1.0, (ntr, nlev, ncells)).astype(np.float32)
    rho_old = rng.uniform(0.8, 1.2, (nlev, ncells)).astype(np.float32)
    ww = (0.4 * rng.normal(size=(nlev + 1, ncells))).astype(np.float32)
    ww[0] = ww[nlev] = 0.0
    dt = np.float32(1.0)
    upwind = rng.normal(size=(ntr, nlev, nedges)).astype(np.float32)
    residual = rng.normal(size=(ntr, nlev, nedges)).astype(np.float32)
    register = np.zeros((ntr, nlev, nslots))
    register[:, :, 1] = rng.uniform(0.0, 3.0, (ntr, nlev))      # the packet draws the column down
    air = np.zeros((nlev, nslots))
    air[:, 1] = rng.uniform(-1.0, 3.0, nlev)                    # dry mass through the packet edge
    upwind[:, :, 2] = register[:, :, 1]
    residual[:, :, 2] = 0.0
    coef = np.array([[-0.5]])                                   # cell 2 is the edge's first cell: -1/area
    # the native vertical low order, as fct_vertical_low_order forms it
    mass = (source_old * rho_old[None]).astype(np.float64)
    vres = rng.normal(size=(ntr, nlev + 1, ncells)).astype(np.float32)
    for i in range(1, nlev):
        flux = dt * (np.maximum(ww[i], 0) * source_old[:, i - 1] + np.minimum(ww[i], 0) * source_old[:, i])
        mass[:, i - 1] -= flux * rdzw[i - 1]
        mass[:, i] += flux * rdzw[i]
    mass = mass.astype(np.float32)
    column_before = (mass[:, :, 2].astype(np.float64) / rdzw[None, :]).sum(axis=1)
    packet_edge = np.zeros(nedges, dtype=np.uint8)
    packet_edge[2] = 1
    d = {name: cp.asarray(value) for name, value in dict(
        upwind=upwind, residual=residual, mass=mass, vres=vres).items()}
    stats = cp.zeros(4, dtype=cp.uint64)
    i32 = np.int32
    _launch(kernels["lts_donor_guard"], ntr * 1, (
        i32(ntr), i32(nlev), i32(1), i32(1), i32(ncells), i32(nedges), i32(nslots), i32(max_edges), dt,
        cp.asarray(np.array([2], np.int32)), cp.asarray(np.array([1], np.int32)), cp.asarray(coef.reshape(-1)),
        cp.asarray(register), cp.asarray(air), cp.asarray(np.full(5, 2, np.int32)),
        cp.asarray(edges_on_cell.reshape(-1)),
        cp.asarray(acoustic_sign.reshape(-1)), cp.asarray(inv_area), cp.asarray(packet_edge),
        cp.asarray(source_old), cp.asarray(rho_old), cp.asarray(ww), cp.asarray(rdzw),
        d["upwind"], d["residual"], d["mass"], d["vres"], stats, cp.zeros(3 * ntr, dtype=cp.float64)))
    new_up, new_mass = cp.asnumpy(d["upwind"]).astype(np.float64), cp.asnumpy(d["mass"]).astype(np.float64)
    assert int(cp.asnumpy(stats)[0]) > 0
    # the packet edge and every other column are untouched
    np.testing.assert_array_equal(new_up[:, :, 2], upwind[:, :, 2])
    np.testing.assert_array_equal(new_mass[:, :, [0, 1, 3, 4]], mass[:, :, [0, 1, 3, 4]])
    # scaling a vertical flux moves mass between two levels of the column and creates none
    column_after = (new_mass[:, :, 2] / rdzw[None, :]).sum(axis=1)
    np.testing.assert_allclose(column_after, column_before, rtol=2e-6)
    scaled = 0
    for t in range(ntr):
        for k in range(nlev):
            value = float(source_old[t, k, 2])
            content = value * float(rho_old[k, 2])
            exported = 0.5 * register[t, k, 1]
            air_out = max(0.5 * air[k, 1], 0.0)
            up_w = max(float(ww[k + 1, 2]), 0.0) if k + 1 < nlev else 0.0
            down_w = min(float(ww[k, 2]), 0.0) if k > 0 else 0.0
            vertical = float(rdzw[k]) * (up_w - down_w) * value
            # edge 1 joins cells 1 and 2: a negative flux leaves cell 2
            horizontal_before = max(-float(upwind[t, k, 1]) * 0.5, 0.0)
            outflow = vertical + horizontal_before
            horizontal_after = max(-new_up[t, k, 1] * 0.5, 0.0)
            need = exported + outflow - content
            native = max(value * air_out + outflow - content, 0.0)
            reduce = need - native
            tolerance = 4.0e-6 * (abs(exported) + outflow + abs(content)) if native > 0.0 else 0.0
            if need > 0.0 and reduce > tolerance and outflow > 0.0:
                scaled += 1
                keep = (outflow - reduce) / outflow if reduce < outflow else 0.0
                assert horizontal_after == pytest.approx(horizontal_before * keep, rel=1e-5, abs=1e-7)
                # what is left covers the overdraw the packets made, and no more
                assert outflow * keep == pytest.approx(max(outflow - reduce, 0.0), rel=1e-12, abs=1e-12)
            else:
                assert new_up[t, k, 1] == upwind[t, k, 1]
    assert scaled == int(cp.asnumpy(stats)[0])
