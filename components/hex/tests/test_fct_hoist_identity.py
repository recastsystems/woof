"""The hoisted FCT kernels write the words of the forms they replace.

``woof.hex.cuda_edge_hoist`` selects, from ``HOIST_MIN_EDGES`` edges up, the hoisted edge-value kernel (one thread per
edge and level chunk) and the element-parallel minimum/maximum and target-density kernels in place of the
one-thread-per-edge and one-thread-per-cell forms.  These cases launch both forms of each on a synthetic irregular
mesh (mixed stencil widths, unequal cell degrees, subnormal and signed-zero inputs) at several levels-per-thread
settings and compare every output word.  Brought from fork jacobbieker/woof PR 224 (Apache-2.0).
"""
from __future__ import annotations

import numpy as np
import pytest

requires_gpu = pytest.mark.skipif(__import__('os').environ.get('HEX_TEST_CUDA') != '1', reason='requires an owned CUDA card')

NLEV, NCELLS, NEDGES, MAX_EDGES, WIDTH, NTRACERS = 13, 211, 617, 7, 10, 3


def _mesh(cp, seed=20261108):
    rng = np.random.default_rng(seed)
    f32, i32 = np.float32, np.int32

    def field(*shape, low=-2.0, high=2.0):
        a = rng.uniform(low, high, size=shape).astype(f32)
        flat = a.reshape(-1)
        flat[:3] = (np.float32(1e-40), np.float32(-0.0), np.float32(0.0))
        return cp.asarray(a)

    n_adv = rng.choice([10, 10, 10, 7, 4], size=NEDGES).astype(i32)
    return dict(
        stage=field(NTRACERS, NLEV, NCELLS),
        velocity=field(NLEV, NEDGES),
        adv=field(NEDGES, WIDTH), adv3=field(NEDGES, WIDTH),
        n_adv=cp.asarray(n_adv),
        adv_cells=cp.asarray(rng.integers(0, NCELLS, size=(NEDGES, WIDTH)).astype(i32)),
        counts=cp.asarray(rng.integers(5, MAX_EDGES + 1, size=NCELLS).astype(i32)),
        neighbors=cp.asarray(rng.integers(0, NCELLS, size=(NCELLS, MAX_EDGES)).astype(i32)),
        edges=cp.asarray(rng.integers(0, NEDGES, size=(NCELLS, MAX_EDGES)).astype(i32)),
        signs=cp.asarray(rng.choice([-1.0, 1.0], size=(NCELLS, MAX_EDGES)).astype(f32)),
        dv=field(NEDGES, low=0.5, high=2.0),
        inv_area=field(NCELLS, low=0.1, high=1.0),
        vertical=field(NLEV + 1, NCELLS),
        rdnw=field(NLEV, low=0.5, high=2.0),
        rho0=field(NLEV, NCELLS, low=0.5, high=1.5),
        rho1=field(NLEV, NCELLS, low=0.5, high=1.5),
        old=field(NTRACERS, NLEV, NCELLS),
        source=field(NTRACERS, NLEV, NCELLS),
    )


def _words(cp, a):
    return cp.asnumpy(cp.ascontiguousarray(a).view(cp.uint32))


@requires_gpu
@pytest.mark.parametrize("lpt", [1, 4, NLEV])
def test_edge_values_hoisted_match_the_edge_form(lpt):
    import cupy as cp
    from woof.hex import cuda_edge_hoist
    from woof.hex.cuda_transport_v841 import _launch

    m = _mesh(cp)
    inputs = (np.int32(NTRACERS), np.int32(NLEV), np.int32(NCELLS), np.int32(NEDGES),
              np.int32(WIDTH), np.float32(0.25), m["stage"], m["velocity"], m["adv"],
              m["adv3"], m["n_adv"], m["adv_cells"])
    element = cp.full((NTRACERS, NLEV, NEDGES), np.nan, dtype=cp.float32)
    hoisted = cp.full_like(element, -7.0)
    _launch("transport_edge_values_mono_v841", NEDGES, (*inputs, element), None)
    _launch("transport_edge_values_mono_hoist_v841",
            NTRACERS * NEDGES * cuda_edge_hoist.chunks(NLEV, lpt),
            (*inputs, hoisted, np.int32(lpt)), None)
    assert np.array_equal(_words(cp, hoisted), _words(cp, element))


@requires_gpu
def test_minmax_elements_match_the_cell_form():
    import cupy as cp
    from woof.hex.cuda_transport_v841 import _launch

    m = _mesh(cp)
    inputs = (np.int32(NTRACERS), np.int32(NLEV), np.int32(NCELLS), np.int32(MAX_EDGES),
              np.float32(30.0), m["counts"], m["neighbors"], m["old"], m["source"], m["rho0"])
    cell = [cp.full_like(m["old"], np.nan) for _ in range(3)]
    element = [cp.full_like(m["old"], -7.0) for _ in range(3)]
    _launch("fct_minmax_source", NCELLS, (*inputs, *cell), None)
    _launch("fct_minmax_source_elem_v841", NTRACERS * NLEV * NCELLS, (*inputs, *element), None)
    for got, expected in zip(element, cell):
        assert np.array_equal(_words(cp, got), _words(cp, expected))


@requires_gpu
@pytest.mark.parametrize("advance_density", [0, 1])
def test_target_density_elements_match_the_cell_form(advance_density):
    import cupy as cp
    from woof.hex.cuda_transport_v841 import _launch

    m = _mesh(cp)
    inputs = (np.int32(NLEV), np.int32(NCELLS), np.int32(NEDGES), np.int32(MAX_EDGES),
              np.int32(advance_density), np.float32(30.0), m["counts"], m["edges"], m["signs"],
              m["dv"], m["inv_area"], m["velocity"], m["vertical"], m["rdnw"], m["rho0"], m["rho1"])
    cell = cp.full_like(m["rho0"], np.nan)
    element = cp.full_like(m["rho0"], -7.0)
    _launch("transport_target_density_v841", NCELLS, (*inputs, cell), None)
    _launch("transport_target_density_elem_v841", NLEV * NCELLS, (*inputs, element), None)
    assert np.array_equal(_words(cp, element), _words(cp, cell))


def test_the_hoist_starts_at_fifteen_thousand_edges_and_the_check_takes_a_settle():
    import inspect

    from woof.hex import cuda_edge_hoist
    from woof.hex.cuda_transport_v841 import _launch_checked

    assert cuda_edge_hoist.HOIST_MIN_EDGES == 15_000
    assert "settle" in inspect.signature(_launch_checked).parameters
