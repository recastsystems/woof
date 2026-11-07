"""An N-rank MPAS run has one wave service, on rank 0, fed by every rank's owned forcing.

THE BREAKAGE THIS CHECKS: the wave service registry is per process and per card, so same-host rank processes,
multi-card ranks and --hosts ranks each built a whole-mesh wave model that saw winds only on its own cells, and
all of them wrote the same waves/ files. Two rank threads with the real rank communicator stand in for any
executor; numpy stands in for CuPy and a recording service for the wave model, so no card is needed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from woof.hex import waves  # noqa: E402
from woof.hex.mpasphys.samehost_v842 import ThreadComm  # noqa: E402


class _FakeCupy(SimpleNamespace):
    pass


_CP = _FakeCupy(asarray=np.asarray, ascontiguousarray=np.ascontiguousarray, stack=np.stack, arange=np.arange,
                int32=np.int32, float32=np.float32, asnumpy=np.asarray, ndarray=np.ndarray, take=np.take)


class _Service:
    def __init__(self, n):
        self.z0 = (np.arange(n, dtype=np.float32) + 1) / 1000
        self.bindings, self.handoffs, self.outputs, self.clock = [], [], [], 0.0

    def bind(self, *, local, global_cells, all_global_cells, label):
        binding = SimpleNamespace(local=local, global_cells=np.asarray(global_cells),
                                  all_global_cells=np.asarray(all_global_cells), label=f"{label}:{len(self.bindings)}")
        binding.roughness = lambda b=binding: self.z0[b.all_global_cells]
        self.bindings.append(binding)
        return binding

    def handoff(self, binding, fields, endpoint):
        self.handoffs.append((binding.label, {k: np.array(v) for k, v in fields.items()}, endpoint))

    def output(self, endpoint):
        self.outputs.append(endpoint)


def _session(values):
    qv = np.zeros((1, 1, values.size), np.float32)
    def field(pool, name):
        if (pool, name) == ("diag", "rho"):
            return values[None] * 0 + 1
        if (pool, name) == ("state", "scalars"):
            return qv
        return values * {"u10": 1, "v10": 2, "surface_pressure": 3, "ust": 4}[name]
    return SimpleNamespace(field=field, layout=SimpleNamespace(indices={("state", "index_qv"): 1}))


@pytest.fixture
def two_ranks(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", _CP)
    monkeypatch.setattr(waves, "configuration", lambda: {"enabled": True, "roughness_feedback": True})
    services = []

    def for_mesh(mesh, **kwargs):
        services.append(_Service(int(mesh.dimensions["nCells"])))
        return services[-1]
    monkeypatch.setattr(waves, "for_mesh", for_mesh)
    mesh = SimpleNamespace(dimensions={"nCells": 6})
    # Rank 0 owns global cells 4, 0, 2 (local 0-2, halo 3); rank 1 owns 1, 5, 3 (local 0-2, halo 3).
    partitions = [SimpleNamespace(cell_l2g=np.array([4, 0, 2, 1]), n_owned_cells=3),
                  SimpleNamespace(cell_l2g=np.array([1, 5, 3, 2]), n_owned_cells=3)]
    comms = ThreadComm.world(2)
    executions = []
    for rank in range(2):
        execution = SimpleNamespace(partitions=partitions, wave_rank_comm=lambda c=comms[rank]: c)
        execution.wave_input = lambda p=partitions[rank]: (mesh, p.cell_l2g, p.n_owned_cells)
        executions.append(execution)
    sessions = [_session(np.array([10, 11, 12, 99], np.float32)), _session(np.array([20, 21, 22, 99], np.float32))]
    return SimpleNamespace(mesh=mesh, partitions=partitions, executions=executions, sessions=sessions,
                           services=services)


def _run(two_ranks):
    start = datetime(2023, 3, 31, 12, tzinfo=timezone.utc)

    def rank(r):
        binding = waves.attach_mpas(two_ranks.sessions[r], two_ranks.mesh, SimpleNamespace(grid=None), start=start,
                                    dt=720.0, steps=5, out="unused", execution=two_ranks.executions[r])
        binding.after_mpas_step(two_ranks.sessions[r], 720.0)
        return binding

    with ThreadPoolExecutor(2) as pool:
        return list(pool.map(rank, range(2)))


def test_rank_zero_owns_the_one_service_and_receives_every_ranks_owned_forcing(two_ranks):
    bindings = _run(two_ranks)
    assert len(two_ranks.services) == 1          # old code: one whole-mesh service built by every rank
    service = two_ranks.services[0]
    assert [b.global_cells.tolist() for b in service.bindings] == [[4, 0, 2], [1, 5, 3]]
    assert [label for label, _, _ in service.handoffs] == ["mpas:0", "mpas:1"]
    assert service.handoffs[0][1]["u10"].tolist() == [10, 11, 12]
    assert service.handoffs[1][1]["psfc"].tolist() == [60, 63, 66]   # rank 1's owned cells only, no halo
    assert all(endpoint == 720.0 for _, _, endpoint in service.handoffs)
    assert service.outputs == [0.0]             # rank 0 writes; rank 1 never touches waves/
    assert bindings[1].service.output(720.0) is None
    assert bindings[1].roughness().tolist() == pytest.approx(service.z0[[1, 5, 3]].tolist())
    assert bindings[0].roughness().tolist() == pytest.approx(service.z0[[4, 0, 2]].tolist())
