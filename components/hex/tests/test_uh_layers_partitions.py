"""Updraft helicity on the partitioned WOOF executors: LTS classes (one card or several) and same-host ranks.

THE BREAKAGE THIS GUARDS (both-routes matrix, woof hex 0.4.14 candidate, 2026-11-06): the WOOF fixed-step driver
wrote updraft_helicity_max, but the full-step LTS class executor and the same-host multi-card ranks never armed the
layers, so those runs rendered no UH maps. Each partition now arms its own layers over its local mesh; a cell sums its
kites in its own verticesOnCell order, so a partition's owned words are the whole mesh's words.

CPU tier: a NumPy stand-in for CuPy whose kernel is the CUDA kernel's loop, float32 and in the same order.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex import uh_layers


def _kernel(grid, block, args):
    n, nl, degree, offsets, entries, kite, vort, vstride, w, wstride, z, zstride, area, acc = args
    n, nl, degree = int(n), int(nl), int(degree)
    bot = np.float32([2000.0, 0.0, 0.0, 0.0])
    top = np.float32([5000.0, 1000.0, 2000.0, 3000.0])
    vort, w, z = vort.reshape(-1), w.reshape(-1), z.reshape(-1)
    acc2 = acc.reshape(4, -1)
    for c in range(n):
        r = np.zeros(4, np.float32)
        z0 = z[c]
        for k in range(nl):
            zl = np.float32(z[k * int(zstride) + c] - z0)
            if zl >= 5000.0:
                break
            zu = np.float32(z[(k + 1) * int(zstride) + c] - z0)
            s = np.float32(0.0)
            for e in range(offsets[c], offsets[c + 1]):
                ix = int(entries[e])
                s = np.float32(s + np.float32(kite[ix] * vort[k * int(vstride) + ix // degree]))
            wm = np.float32(0.5) * np.float32(w[k * int(wstride) + c] + w[(k + 1) * int(wstride) + c])
            h = np.float32(max(wm, np.float32(0.0)) * max(np.float32(s / area[c]), np.float32(0.0)))
            for b in range(4):
                zb, zt = max(zl, bot[b]), min(zu, top[b])
                r[b] = np.float32(r[b] + np.float32(h * max(np.float32(0.0), np.float32(zt - zb))))
        acc2[:, c] = np.maximum(r, acc2[:, c])


class _Cp:
    float32 = np.float32
    asarray = staticmethod(np.asarray)
    zeros = staticmethod(np.zeros)
    asnumpy = staticmethod(np.array)
    ascontiguousarray = staticmethod(np.ascontiguousarray)

    class RawKernel:
        def __init__(self, source, name):
            pass

        def __call__(self, grid, block, args):
            _kernel(grid, block, args)


# Four cells around a strip of six vertices; every cell has three vertices (degree-3 rows, sentinel 4 = no cell).
VOC = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4], [3, 5, 4]])
COV = np.array([[0, 4, 4], [0, 1, 4], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 4]])
# Kites and vorticities whose float32 sum depends on its order (1e8 + 1 - 1e8).
KITE = np.array([[1e8, 0, 0], [1.0, -1e8, 0], [-1e8, 1.0, 3.0], [1e8, 2.0, -1e8], [1.0, 1e8, 0], [1.0, 0, 0]],
                np.float32)
NL = 4
VORT = np.tile(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], np.float32), (NL, 1))
W = np.full((NL + 1, 4), 2.0, np.float32)
ZGRID = np.tile(np.array([0.0, 1500.0, 3000.0, 4500.0, 6000.0], np.float32)[:, None], (1, 4))


def _mesh(voc, cov, kite):
    return SimpleNamespace(arrays={"cellsOnVertex": cov, "kiteAreasOnVertex": kite, "verticesOnCell": voc,
                                   "nEdgesOnCell": np.full(voc.shape[0], 3), "areaCell": np.ones(voc.shape[0],
                                                                                                  np.float32)})


def _driver(n, vort, w, z):
    return SimpleNamespace(
        horizontal=SimpleNamespace(ncells=n, solve_vorticity=lambda rho, rho_u, normal_velocity=None: vort),
        atmosphere=SimpleNamespace(state=SimpleNamespace(rho=None, rho_u=None),
                                   saved=SimpleNamespace(normal_velocity=None, vertical_velocity=w),
                                   vertical=SimpleNamespace(zgrid=z)))


def _local(order):
    """The same four cells with the vertices renumbered in ``order`` (local vertex i is global order[i])."""
    g2l = {g: i for i, g in enumerate(order)}
    voc = np.vectorize(g2l.get)(VOC)
    cov = COV[order]
    return voc, cov, KITE[order], VORT[:, order]


@pytest.fixture(autouse=True)
def _default_layers(monkeypatch):
    monkeypatch.delenv(uh_layers.ENV, raising=False)


def test_a_partition_with_renumbered_vertices_writes_the_whole_meshs_words():
    whole = {"driver": _driver(4, VORT, W, ZGRID)}
    assert uh_layers.attach_rank(whole, _mesh(VOC, COV, KITE), _Cp(), initial=True) is not None
    reference = uh_layers.rank_history_arrays(whole)["updraft_helicity_max"]
    assert reference.any()
    voc, cov, kite, vort = _local([0, 2, 1, 3, 4, 5])
    part = {"driver": _driver(4, vort, W, ZGRID)}
    uh_layers.attach_rank(part, _mesh(voc, cov, kite), _Cp(), initial=True)
    words = uh_layers.rank_history_arrays(part)["updraft_helicity_max"]
    assert words.view(np.uint32).tolist() == reference.view(np.uint32).tolist()


def test_a_partition_folds_after_its_steps_and_resets_at_each_frame():
    stack = {"driver": _driver(4, VORT, W, ZGRID)}
    uh_layers.attach_rank(stack, _mesh(VOC, COV, KITE), _Cp(), initial=False)
    assert not uh_layers.rank_history_arrays(stack)["updraft_helicity_max"].any()   # nothing folded yet
    uh_layers.rank_update(stack)
    first = uh_layers.rank_history_arrays(stack)["updraft_helicity_max"]
    assert first.any()
    assert not uh_layers.rank_history_arrays(stack)["updraft_helicity_max"].any()   # reset at the frame


def test_a_limited_area_partition_writes_its_garbage_column_too():
    stack = {"driver": _driver(5, VORT, np.full((NL + 1, 5), 2.0, np.float32),
                               np.tile(np.array([0.0, 1500.0, 3000.0, 4500.0, 6000.0], np.float32)[:, None], (1, 5)))}
    uh_layers.attach_rank(stack, _mesh(VOC, COV, KITE), _Cp(), initial=True)
    assert uh_layers.rank_history_arrays(stack)["updraft_helicity_max"].shape == (5,)


def test_a_faulted_partition_hands_zeros_under_every_name(monkeypatch):
    monkeypatch.setenv(uh_layers.ENV, "1")
    stack = {"driver": _driver(4, VORT, W, ZGRID)}
    uh_layers.attach_rank(stack, _mesh(VOC, COV, KITE), _Cp(), initial=False)
    stack["uh_layers"].fail("test", RuntimeError("x"))
    arrays = uh_layers.rank_history_arrays(stack)
    assert tuple(arrays) == uh_layers.NAMES and not any(a.any() for a in arrays.values())


def test_turned_off_no_partition_arms_the_layers(monkeypatch):
    monkeypatch.setenv(uh_layers.ENV, "off")
    stack = {"driver": _driver(4, VORT, W, ZGRID)}
    assert uh_layers.attach_rank(stack, _mesh(VOC, COV, KITE), _Cp(), initial=True) is None
    assert uh_layers.rank_history_arrays(stack) == {}


def test_every_partitioned_executor_arms_folds_and_captures_the_layers():
    drivers = Path(uh_layers.__file__).with_name("drivers")
    lts = drivers.joinpath("fullstep_lts_forecast_v841.py").read_text()
    assert lts.count("attach_rank(stack, local_host") == 2      # one card and multi-card class executors
    assert lts.count("rank_update(stack)") == 2
    ranks = drivers.joinpath("samehost_rank_group_v841.py").read_text()
    assert "attach_rank(stack, local[" in ranks and "rank_update(stack)" in ranks
    capture = drivers.joinpath("samehost_forecast_v841.py").read_text()
    assert "arrays.update(rank_history_arrays(stack))" in capture
