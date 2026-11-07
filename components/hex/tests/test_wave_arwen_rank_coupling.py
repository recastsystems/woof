"""ArWen-route rank groups have one wave service, on rank 0, fed by every rank's owned forcing.

THE BREAKAGE THIS CHECKS: attach_arwen with a rank layout built a whole-mesh wave service in every rank process
(winds only on that rank's cells) and every rank wrote the same waves/ files. numpy stands in for CuPy and a
recording service for the wave model, so no card is needed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from woof.hex import waves  # noqa: E402

_CP = SimpleNamespace(asarray=np.asarray, ascontiguousarray=np.ascontiguousarray, stack=np.stack, arange=np.arange,
                      concatenate=np.concatenate, int32=np.int32, float32=np.float32, asnumpy=np.asarray)


class _Service:
    def __init__(self, n):
        self.n = n
        self.z0 = (np.arange(n, dtype=np.float32) + 1) / 1000
        self.options = {"roughness_feedback": True}
        self.bindings, self.handoffs, self.outputs, self.clock = [], [], [], 0.0

    def bind(self, *, local, global_cells, all_global_cells, label):
        binding = SimpleNamespace(local=np.asarray(local), global_cells=np.asarray(global_cells),
                                  all_global_cells=np.asarray(all_global_cells),
                                  label=f"{label}:{len(self.bindings)}")
        binding.roughness = lambda b=binding: self.z0[b.all_global_cells]
        binding.service = self
        self.bindings.append(binding)
        return binding

    def handoff(self, binding, fields, endpoint):
        self.handoffs.append((binding.label, {k: np.array(v) for k, v in fields.items()}, endpoint))

    def output(self, endpoint):
        self.outputs.append(endpoint)

    def receipt(self):
        return {"enabled": True}


def _backend(values):
    n = values.size
    fields = {"u10": values.copy(), "v10": values * 2, "psfc": values * 3, "ust": values * 4}
    driver = SimpleNamespace(fields=fields, ocean=None)
    return SimpleNamespace(_constructor=SimpleNamespace(n_columns=n), _seam=SimpleNamespace(_driver=driver,
                           elapsed_seconds=300.0),
                           _les_columns=SimpleNamespace(wave_rho_air=np.ones(n, np.float32)))


def test_rank_zero_owns_the_one_arwen_service_and_receives_the_other_ranks_forcing(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", _CP)
    monkeypatch.setattr(waves, "configuration", lambda: {"enabled": True, "roughness_feedback": True})
    services = []

    def for_mesh(mesh, **kwargs):
        services.append(_Service(int(mesh.dimensions["nCells"])))
        return services[-1]
    monkeypatch.setattr(waves, "for_mesh", for_mesh)
    mesh = SimpleNamespace(dimensions={"nCells": 6}, attrs={})
    host = {"prepared": SimpleNamespace(mesh=mesh)}
    schedule = {"start_time": "2023-03-31T12:00:00", "steps": 12}
    layouts = [SimpleNamespace(cell_l2g=np.array([4, 0, 2, 1]), n_owned_cells=3, n_local_cells=4),
               SimpleNamespace(cell_l2g=np.array([1, 5, 3, 2]), n_owned_cells=3, n_local_cells=4)]
    backends = [_backend(np.array([10, 11, 12, 99], np.float32)), _backend(np.array([20, 21, 22, 99], np.float32))]
    for rank in (1, 0):   # a rank process may attach before rank 0
        waves.attach_arwen(backends[rank], host, schedule, "unused", dt=300.0, layout=layouts[rank],
                           ranks=(rank, layouts))
    assert len(services) == 1                    # old code: every rank built its own service
    service = services[0]
    owner = backends[0].wave_binding
    remote = backends[1].wave_binding
    assert [b.global_cells.tolist() for b in service.bindings] == [[4, 0, 2], [1, 5, 3]]
    assert service.bindings[1].all_global_cells.tolist() == [1, 5, 3, 2]   # rank 1's halo column too
    assert remote.service.output(300.0) is None and service.outputs == [0.0]
    remote.after_arwen_step(backends[1])
    values = waves.couple_rank_group(owner, [remote.take()])
    label, fields, endpoint = service.handoffs[-1]
    assert label == "arwen:1" and endpoint == 300.0
    assert fields["u10"].tolist() == [20, 21, 22] and fields["psfc"].tolist() == [60, 63, 66]
    remote.set_roughness(values[0])
    assert remote.roughness().tolist() == pytest.approx(service.z0[[1, 5, 3, 2]].tolist())
    assert remote.take() is None                 # one step's forcing is handed over once
