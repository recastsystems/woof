"""Scheduler/carrier regressions with a test-only NumPy device shim.

CUDA formulas and global halo indexing are checked in stochastic_forcing.cpp.
These tests do not claim forecast or device execution evidence.
"""
from copy import deepcopy
from types import SimpleNamespace as NS
import sys
import numpy as np
import pytest
from woof.hex import stochastic


def runtime(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cupy', NS(**{**np.__dict__, 'asnumpy': lambda a: a.copy()}))
    class Process:
        def __init__(self):
            self.base = NS(draw=0, last_time=None)
        def advance(self, seconds):
            assert self.base.last_time is None or seconds >= self.base.last_time
            self.base.last_time = seconds
            self.base.draw += 1
            return (np.array([.1,.2,.3,.4,.5], np.float32),
                    np.array([0,0,.1,-.2,.3], np.float32),
                    np.array([1,-1,0,0,0], np.float32))
        def snapshot(self, *, device):
            return deepcopy(vars(self.base))
        def restore(self, value):
            self.base = NS(**deepcopy(value))
    def physics(mesh, vertical, config):
        # A runtime cache on this private view must never enter the host carrier.
        mesh._stochastic_harmonics = {'runtime': object()}
        resource = NS(cells=2, edges=3, coe=np.array([[0,1]]*3, np.int32),
            voe=np.array([[0,1],[1,2],[2,0]], np.int32),
            orientation=np.ones(3, np.float32), dv=np.ones(3, np.float32),
            edge_weight=np.array([.2,.3,.5], np.float32),
            area=np.array([.4,.6], np.float32), receipt={})
        return NS(config=stochastic.StochasticConfig.parse(config), resource=resource,
            process=Process(), taper=np.ones((2,2), np.float32),
            vertex_taper=np.ones((2,3), np.float32), skeb_norm=np.ones(1, np.float32))
    monkeypatch.setattr(stochastic, 'StochasticPhysics', physics)
    def kernel(name):
        def launch(grid, block, args):
            if name == 'lts_sink_cells':
                integral, sink, eoc, nec, weights, nc, ne, max_edges, levels, duration = args
                for cell in range(nc):
                    edges = eoc[cell, :nec[cell]]
                    sink[:,cell] = (integral[:,edges]*weights[edges]).sum(axis=1)/weights[edges].sum()/duration
        return launch
    monkeypatch.setattr(stochastic, '_kernel', kernel)
    mesh = NS(n_cells=2, n_edges=3, arrays={
        'edgesOnCell': np.array([[0,1,2],[2,1,0]], np.int32),
        'nEdgesOnCell': np.array([3,3], np.int32)})
    layouts = [NS(cell_l2g=np.array([1,0]), edge_l2g=np.array([2,0,1]), n_owned_edges=2),
               NS(cell_l2g=np.array([0,1]), edge_l2g=np.array([1,0,2]), n_owned_edges=1)]
    obj = stochastic.LtsStochasticPhysics(mesh, NS(n_vert_levels=2),
                                         dict(sppt=True, skeb=True), layouts)
    assert not hasattr(mesh, '_stochastic_harmonics')
    return obj


def apply(obj, index, seconds, dt):
    momentum = np.full((2,3), -200, np.float32)
    momentum[:, :obj.layouts[index]['owned_edges']] = -2
    held = NS(rho_theta=np.ones((2,2), np.float32), rho_u=momentum)
    obj.apply(index, held, state=NS(rho=np.ones((2,2), np.float32)),
        rho_edge=np.ones((2,3), np.float32), normal_velocity=np.ones((2,3), np.float32),
        time_seconds=seconds, dt=dt, names=())


def test_one_clock_fine_first_owned_budget_and_mesh_independent_restore(monkeypatch):
    obj = runtime(monkeypatch)
    obj.begin_macro(0,4)
    # Fine-first finishes later fine steps before visiting the coarse clock 0.
    for seconds in range(4):
        apply(obj, 0, seconds, 1)
    apply(obj, 1, 0, 4)
    assert obj.process.base.draw == 1
    with pytest.raises(ValueError, match='completed macro boundary'):
        obj.snapshot()
    obj.end_macro()
    # Halo tendencies are 100x larger and must not enter the global budget.
    np.testing.assert_array_equal(obj.dissipation, [[2],[2]])
    saved = obj.snapshot(device=True)
    obj.dissipation[:] = 99
    resumed = runtime(monkeypatch)
    resumed.restore(saved)
    np.testing.assert_array_equal(resumed.dissipation, [[2],[2]])
    assert resumed.boundary_seconds == 4 and resumed.macro_calls == 1
    resumed.begin_macro(4,4)
    assert resumed.process.base.draw == 2
    assert resumed.process.base.last_time == 4
    with pytest.raises(ValueError, match='straddles'):
        apply(resumed, 0, 5, 1)  # Missing first owned interval.
    with pytest.raises(ValueError, match='missing an owned class interval'):
        resumed.end_macro()


@pytest.mark.parametrize('change', [dict(macro_calls=-1), dict(boundary_seconds=None),
    dict(dissipation=np.array([[np.nan],[0]], np.float32)),
    dict(dissipation=np.zeros((2,1), np.float64))])
def test_invalid_boundary_carrier_is_refused(monkeypatch, change):
    obj = runtime(monkeypatch)
    obj.begin_macro(0,4)
    apply(obj,0,0,4)
    apply(obj,1,0,4)
    obj.end_macro()
    saved = obj.snapshot()
    saved.update(change)
    with pytest.raises(ValueError, match='checkpoint'):
        runtime(monkeypatch).restore(saved)
