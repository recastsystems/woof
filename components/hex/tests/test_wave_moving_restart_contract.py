"""CPU orchestration check, with an identity overlap and stub device kernel.

This exercises the actual moving restore method, not GPU transport accuracy.
"""
import copy
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix


def fixture(monkeypatch):
    from woof.hex.waves import HexWaveService

    monkeypatch.setitem(sys.modules, "cupyx.scipy.sparse", SimpleNamespace(csr_matrix=csr_matrix))
    arrays = dict(action=np.ones((2, 4, 4), np.float32),
                  depth=np.full(2, 20, np.float32), water_level=np.zeros(2, np.float32),
                  depth_previous=np.full(2, 20, np.float32),
                  surface_iteration=np.zeros((2, 12), np.float32),
                  shelter_iteration=np.zeros((2, 4, 2), np.float32),
                  wind_input_iteration=np.full((2, 1), 0xA55A, np.uint32),
                  clock=np.array([600.]), steps=np.array([2], np.int64))
    metadata = dict(source_parameter_sha256="source", partition_shapes_sha256="shapes")
    events = []

    class Solver:
        def restart_state(self):
            return dict(metadata=metadata.copy(), arrays=copy.deepcopy(arrays))

        def restore_state(self, payload):
            self.arrays = payload["arrays"]
            assert self.arrays["wind_input_iteration"].dtype == np.uint32
            events.append("restore")

        def set_forcing(self, forcing):
            self.forcing = forcing
            events.append("forcing")

        def refresh_outputs(self, *, refresh_stress):
            assert refresh_stress
            np.testing.assert_array_equal(self.forcing.u10, [7., 9.])
            np.testing.assert_array_equal(self.forcing.v10, [2., 3.])
            assert self.forcing.time_seconds == 600.
            self.arrays["wind_input_iteration"].fill(0xFFFF)
            events.append("diagnostics")

    service = object.__new__(HexWaveService)
    service.cp = np
    service.options = {}
    service.start_epoch, service.initial_seconds = 0., 600.
    service.solver = Solver()
    service.mesh = SimpleNamespace(arrays=dict(latCell=np.zeros(2), lonCell=np.zeros(2)))
    service.backend = SimpleNamespace(ncell=2, wet_to_full=np.arange(2, dtype=np.int32))
    service.bathy = SimpleNamespace(depth=np.full(2, 20, np.float32))
    service.buffers = {key: np.zeros(2, np.float32) for key in
                       ("u10", "v10", "tau_x", "tau_y", "psfc", "ustar", "rho_air")}
    service.refresh_roughness = lambda: events.append("feedback")
    service.restore_point_frames = lambda _: events.append("stations")
    service.remap_kernel = lambda grid, block, args: np.copyto(args[2], args[0])
    forcing = {key: np.array([1., 1.], np.float32) for key in service.buffers}
    forcing.update(u10=np.array([7., 9.], np.float32), v10=np.array([2., 3.], np.float32))
    state = dict(options={}, start_epoch=0., clock=600., solver=service.solver.restart_state(),
                 forcing=forcing, wet_to_full=np.arange(2, dtype=np.int32),
                 full_to_wet=np.arange(2, dtype=np.int32))
    weights = SimpleNamespace(overlap=csr_matrix(np.eye(2)), target_area=np.ones(2))
    return service, state, weights, events


def test_moving_rebuilds_packed_mask_after_mapped_forcing(monkeypatch):
    service, state, weights, events = fixture(monkeypatch)
    service.restore_moving(state, service.mesh, weights)
    assert events == ["restore", "forcing", "diagnostics", "feedback", "stations"]
    np.testing.assert_array_equal(service.solver.arrays["wind_input_iteration"], [[0xFFFF], [0xFFFF]])
    np.testing.assert_array_equal(state["solver"]["arrays"]["wind_input_iteration"], [[0xA55A], [0xA55A]])


def test_moving_still_refuses_undeclared_core_state(monkeypatch):
    service, state, weights, events = fixture(monkeypatch)
    restart = service.solver.restart_state

    def changed():
        payload = restart()
        payload["arrays"]["new_prognostic_state"] = np.zeros(2)
        return payload

    service.solver.restart_state = changed
    with pytest.raises(ValueError, match="declared rule"):
        service.restore_moving(state, service.mesh, weights)
    assert not events


def test_waves_off_attach_accepts_consumed_lts_carrier(monkeypatch, tmp_path):
    from woof.hex import waves

    monkeypatch.setattr(waves, "configuration", lambda: {"enabled": False})
    # The real LTS spent carrier has already released its consumed mesh.
    spent = {"time_seconds": 1800., "waves": None}
    assert waves.attach_mpas(None, None, None, start=None, dt=1., steps=1,
                             out=tmp_path, carried=None, weights=object(),
                             source_carrier=spent) is None


def test_carried_waves_still_restore_from_the_source_mesh(monkeypatch, tmp_path):
    from woof.hex import waves

    monkeypatch.setattr(waves, "configuration", lambda: {"enabled": True})
    monkeypatch.setitem(sys.modules, "cupy", np)
    source, target, weights = object(), object(), object()
    carried = {"start_epoch": 0.}
    seen = []
    binding = object()
    service = SimpleNamespace(
        clock=600., bind=lambda **kw: binding, on_card=nullcontext,
        restore_moving=lambda state, mesh, remap: seen.append((state, mesh, remap)),
        output=lambda clock: seen.append(clock))
    monkeypatch.setattr(waves, "for_mesh", lambda *args, **kwargs: service)
    execution = SimpleNamespace(wave_input=lambda: (target, np.arange(2), 2))
    result = waves.attach_mpas(SimpleNamespace(), target, SimpleNamespace(grid=None),
        start=None, dt=1., steps=1, out=tmp_path, execution=execution,
        carried=carried, weights=weights, source_carrier={"mesh": source}, clock_offset=600.)
    assert result is binding
    assert seen == [(carried, source, weights), 600.]
