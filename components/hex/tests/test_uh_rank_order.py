"""Updraft helicity on N same-host ranks holds one card's words (WOOF_HEX_UH_RANK_ORDER, 20 Nov 2026).

THE BREAKAGE THIS GUARDS: a rank's helicity gather added each cell's vertex terms in a native rank's own vertex
order, so seam cells summed in another order than one card and the updraft-helicity maxima and minima differed
in the last place there (about 1,300 of 12 million diag words a frame, 1 against 2 ranks, 15 Mar 2025 north
Alabama 3 km circle). The rank here gathers, splits and smooths helicity in one card's order through the module's
own kernels, and every owned cell's words equal one card's.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasphys.rrfs.output_diagnostics_rrfs import (
    native_rank_vertex_order, one_card_rank_vertex_order, one_card_vertex_order, uh_rank_order)

N, NL = 12, 6
NEIGH = ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))      # axial, counter-clockwise


def lattice():
    """A periodic triangular lattice of hexagonal cells (canonical zero-based connectivity, as hexcore's Mesh)."""
    cid = lambda i, j: (i % N) + N * (j % N)
    tris = []
    for j in range(N):
        for i in range(N):
            tris.append((cid(i, j), cid(i + 1, j), cid(i, j + 1)))             # up
            tris.append((cid(i + 1, j), cid(i + 1, j + 1), cid(i, j + 1)))     # down
    cov = np.asarray(tris, dtype=np.int64)
    offs = {}
    for v, (j, i) in enumerate((t // 2 // N, t // 2 % N) for t in range(2 * N * N)):
        up = v % 2 == 0
        cells = ((0, 0), (1, 0), (0, 1)) if up else ((1, 0), (1, 1), (0, 1))
        for (di, dj), c in zip(cells, cov[v]):
            others = [np.subtract(o, (di, dj)) for o in cells if o != (di, dj)]
            centroid = (others[0] + others[1]) / 3.0
            x, y = centroid[0] + 0.5 * centroid[1], centroid[1] * np.sqrt(3) / 2
            offs.setdefault(int(c), []).append((np.arctan2(y, x), v))
    voc = np.asarray([[v for _, v in sorted(offs[c])] for c in range(N * N)], dtype=np.int64)
    edge = {}
    coc = np.zeros((N * N, 6), dtype=np.int64); eoc = np.zeros_like(coc)
    for c in range(N * N):
        i, j = c % N, c // N
        for s, (di, dj) in enumerate(NEIGH):
            o = cid(i + di, j + dj); coc[c, s] = o
            eoc[c, s] = edge.setdefault(frozenset((c, o)), len(edge))
    return SimpleNamespace(verticesOnCell=voc, nEdgesOnCell=np.full(N * N, 6, dtype=np.int64), cellsOnVertex=cov,
                           cellsOnCell=coc, edgesOnCell=eoc, n_edges=len(edge))


def rank_cut(mesh, owned, rng):
    """One rank's local mesh: owned cells (ascending id), halo layers 1-2 (ascending id each), shuffled vertices."""
    layers, have = [np.asarray(sorted(owned))], set(owned)
    for _ in range(2):
        ring = sorted({int(o) for c in layers[-1] for o in mesh.cellsOnCell[c]} - have)
        layers.append(np.asarray(ring)); have |= set(ring)
    cell_l2g = np.concatenate(layers)
    vertex_l2g = np.unique(mesh.verticesOnCell[cell_l2g].ravel())
    vertex_l2g = vertex_l2g[rng.permutation(vertex_l2g.size)]                  # any local vertex numbering
    cl = {int(g): k for k, g in enumerate(cell_l2g)}; vl = {int(g): k for k, g in enumerate(vertex_l2g)}
    nc = cell_l2g.size
    return SimpleNamespace(
        cell_l2g=cell_l2g, vertex_l2g=vertex_l2g, owned=len(owned), nc=nc,
        ends=np.cumsum([len(x) for x in layers]),
        cov1=np.asarray([[cl.get(int(c), nc) + 1 for c in mesh.cellsOnVertex[g]] for g in vertex_l2g], dtype=np.int32),
        voc=np.asarray([[vl[int(v)] for v in mesh.verticesOnCell[g]] for g in cell_l2g], dtype=np.int64),
        coc1=np.asarray([[cl.get(int(o), nc) + 1 for o in mesh.cellsOnCell[g]] for g in cell_l2g], dtype=np.int32))


def helicity(cp, cov1, order, kite, vort, w, area, eoc1, coc1, dv, dc, nc, ns):
    """The module's own kernels: gather (in ``order``), split, smooth; returns the smoothed (positive, negative)."""
    from woof.hex.mpasdiag.convective_rrfs import _module
    from woof.hex.mpasdiag.convective_v842 import helicity_plan
    f = _module(cp.cuda.Device().id).get_function
    offsets, entries = helicity_plan(cp.asarray(cov1), nc, None if order is None else cp.asarray(order))
    uh, neg, spos, sneg = (cp.zeros((nc + 1, NL), dtype=cp.float32) for _ in range(4))
    vort = cp.asarray(np.ascontiguousarray(vort)); w = cp.asarray(np.ascontiguousarray(w))   # (level, *) rows
    launch = lambda name, n, args: f(name)(((n + 127) // 128,), (128,), args)
    launch('vertex_helicity_gather_pool', (nc + 1) * NL, (np.int32(nc), np.int32(NL), np.int32(3), offsets, entries,
           cp.asarray(kite), vort, np.int64(vort.shape[1]), uh))
    launch('split_helicity_pool', nc + 1, (np.int32(nc), np.int32(NL), w, np.int64(w.shape[1]), cp.asarray(area), uh, neg))
    launch('smooth_helicity', ns, (np.int32(ns), np.int32(NL), np.int32(6), cp.zeros(nc, dtype=cp.int32),
           cp.full(nc, 6, dtype=cp.int32), cp.asarray(eoc1), cp.asarray(coc1), cp.asarray(dv), cp.asarray(dc),
           uh, neg, spos, sneg))
    return cp.asnumpy(spos), cp.asnumpy(sneg)


def test_words_default_and_refusal(monkeypatch):
    monkeypatch.delenv('WOOF_HEX_UH_RANK_ORDER', raising=False)
    assert uh_rank_order() == 'one-card'
    monkeypatch.setenv('WOOF_HEX_UH_RANK_ORDER', 'native')
    assert uh_rank_order() == 'native'
    monkeypatch.setenv('WOOF_HEX_UH_RANK_ORDER', 'natve')
    with pytest.raises(ValueError):
        uh_rank_order()


def test_one_card_order_is_the_one_card_session_order():
    """one_card_vertex_order on the global mesh = native_rank_vertex_order on one card's session (pool layout)."""
    mesh = lattice()
    fields = {'verticesOnCell': mesh.verticesOnCell.T + 1, 'nEdgesOnCell': mesh.nEdgesOnCell,
              'cellsOnVertex': mesh.cellsOnVertex.T + 1, 'indexToCellID': np.arange(1, N * N + 1)}
    session = SimpleNamespace(native=SimpleNamespace(get_field=lambda pool, name: fields[name]),
                              layout=SimpleNamespace(dimensions={'nCellsSolve': N * N, 'nCells': N * N,
                                                                 'nVertices': 2 * N * N}),
                              device_pools=SimpleNamespace(cp=np))
    np.testing.assert_array_equal(one_card_vertex_order(mesh, limited_area=True), native_rank_vertex_order(session))
    np.testing.assert_array_equal(one_card_vertex_order(mesh, limited_area=False), np.arange(2 * N * N))


def test_two_ranks_write_one_cards_helicity_words():
    cp = pytest.importorskip('cupy')
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:  # noqa: BLE001
        pytest.skip('no CUDA device')
    rng = np.random.default_rng(20261120)
    mesh = lattice(); nc, nv = N * N, 2 * N * N
    # Wide-range words, so the order of a cell's six vertex terms shows in the last place.
    kite = (rng.uniform(0.2, 1.0, (nv, 3)) * 10.0 ** rng.integers(4, 8, (nv, 3))).astype(np.float32)
    vort = (rng.standard_normal((NL, nv)) * 10.0 ** rng.integers(-5, -1, (NL, nv))).astype(np.float32)
    w = rng.uniform(-3, 12, (NL + 1, nc)).astype(np.float32)
    area = rng.uniform(5e6, 9e6, nc).astype(np.float32)
    dv = rng.uniform(1e3, 3e3, mesh.n_edges).astype(np.float32); dc = rng.uniform(2e3, 4e3, mesh.n_edges).astype(np.float32)
    G = one_card_vertex_order(mesh, limited_area=True)
    one = helicity(cp, mesh.cellsOnVertex.astype(np.int32) + 1, G, kite, vort, w, area,
                   mesh.edgesOnCell.astype(np.int32) + 1, mesh.cellsOnCell.astype(np.int32) + 1, dv, dc, nc, nc)
    differ = {'one-card': 0, 'native': 0}
    for owned in (range(0, nc // 2), range(nc // 2, nc)):                      # two ranks: rows 0-5, rows 6-11
        r = rank_cut(mesh, list(owned), rng)
        g = r.cell_l2g
        for word in differ:
            if word == 'one-card':
                order = one_card_rank_vertex_order(r.vertex_l2g, G)
            else:
                fields = {'verticesOnCell': r.voc.T + 1, 'nEdgesOnCell': np.full(r.nc, 6),
                          'cellsOnVertex': r.cov1.T.astype(np.int64), 'indexToCellID': g + 1}
                order = native_rank_vertex_order(SimpleNamespace(
                    native=SimpleNamespace(get_field=lambda pool, name: fields[name]),
                    layout=SimpleNamespace(dimensions={'nCellsSolve': r.owned, 'nCells': r.nc, 'nVertices': r.vertex_l2g.size},
                                           halo_ends={'nCells': tuple(int(e) for e in r.ends)}),
                    device_pools=SimpleNamespace(cp=np)))
            pos, neg = helicity(cp, r.cov1, order, kite[r.vertex_l2g], vort[:, r.vertex_l2g], w[:, g], area[g],
                                mesh.edgesOnCell[g].astype(np.int32) + 1, r.coc1, dv, dc, r.nc, r.owned)
            o = slice(0, r.owned)
            differ[word] += int(np.count_nonzero(pos[o].view(np.uint32) != one[0][g[o]].view(np.uint32)))
            differ[word] += int(np.count_nonzero(neg[o].view(np.uint32) != one[1][g[o]].view(np.uint32)))
    assert differ['one-card'] == 0, differ
    # The old order on the same ranks: the seam cells' words differ (what the default no longer does).
    assert differ['native'] > 0, differ
