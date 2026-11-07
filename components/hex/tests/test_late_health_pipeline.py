"""Late health preserves both boundaries, carry and ordered receipts."""
from dataclasses import dataclass
import json
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from woof.hex.cuda_seam_snapshot_v841 import TwoDeepForecastSnapshotV841
from woof.hex.step_receipt_writer import StepReceiptWriter


@dataclass(frozen=True)
class Carrier:
    values: object
    time_seconds: float = 0.0

    @property
    def rho(self):
        return self.values


@dataclass(frozen=True)
class Atmosphere:
    state: Carrier
    saved: Carrier


class Seam:
    def __init__(self, cp, values):
        self._cp, self.values = cp, values
        self.step = 0

    def export_state(self):
        return {"arrays": {"values": self._cp.asnumpy(self.values)},
                "scalars": {"step": self.step}, "identity": {"schema": 1}}


class Backend:
    def __init__(self, cp):
        self.cp = cp
        self._seam = Seam(cp, cp.array([-0.0, 1.0], dtype=np.float32))
        self._phase = "boundary"
        self._gwdo_calls = 0
        self._gf_forcing_seen = False
        self._last_gwdo_result = Carrier(cp.array([2.0]))
        self._committed_refl10cm = cp.array([3.0])
        self._last_receipt = {"step": 0}

    def _restore_boundary(self, payload):
        # Mirror the adapter's consuming public restore contract.
        self._seam = Seam(self.cp, self.cp.array(payload["arrays"]["values"], copy=True))
        self._seam.step = payload["scalars"]["step"]
        payload["arrays"].clear()
        self._phase = "boundary"


class CPUArrays:
    ndarray = np.ndarray
    array = staticmethod(np.array)
    asnumpy = staticmethod(np.asarray)
    # Stands in for cupy: the backend selector reads cupy.cuda.runtime.is_hip.
    cuda = SimpleNamespace(runtime=SimpleNamespace(is_hip=False))


def exercise_two_boundaries(cp):
    backend = Backend(cp)
    atmosphere = Atmosphere(Carrier(cp.array([-0.0, 4.0])), Carrier(cp.array([5.0])))
    driver = SimpleNamespace(atmosphere=atmosphere)
    snapshots = TwoDeepForecastSnapshotV841({"driver": driver, "backend": backend}, cp)
    surface, gf = {"t2": cp.array([6.0])}, Carrier(cp.array([7.0]))
    snapshots.capture(0, surface, gf)
    driver.atmosphere = Atmosphere(Carrier(cp.array([10.0, 20.0]), 5.0), Carrier(cp.array([11.0])))
    backend._seam.values.fill(12)
    backend._seam.step = 1
    backend._gwdo_calls = 1
    backend._gf_forcing_seen = True
    backend._last_receipt = {"step": 1}
    surface["t2"].fill(13)
    gf.values.fill(14)
    snapshots.capture(1, surface, gf)
    before = snapshots.receipt()
    buffers = {key: id(value) for key, value in snapshots.slots[0]["copies"].buffers.items()}
    # Capturing boundary 2 reuses slot 0 without destroying boundary 1.
    snapshots.capture(2, surface, gf)
    assert buffers == {key: id(value) for key, value in snapshots.slots[0]["copies"].buffers.items()}
    driver.atmosphere.state.values.fill(999)
    backend._seam.values.fill(999)
    backend._last_gwdo_result.values.fill(999)
    backend._committed_refl10cm.fill(999)
    backend._gwdo_calls = 999
    surface["t2"].fill(999)
    gf.values.fill(999)
    restored_surface, restored_gf = snapshots.restore(1)
    assert cp.asnumpy(driver.atmosphere.state.values).tolist() == [10.0, 20.0]
    assert driver.atmosphere.state.time_seconds == 5.0
    assert cp.asnumpy(driver.atmosphere.saved.values).tolist() == [11.0]
    assert cp.asnumpy(backend._seam.values).tolist() == [12.0, 12.0]
    assert backend._seam.step == 1 and backend._gwdo_calls == 1
    assert backend._gf_forcing_seen and backend._last_receipt == {"step": 1}
    assert cp.asnumpy(backend._last_gwdo_result.values).tolist() == [2.0]
    assert cp.asnumpy(backend._committed_refl10cm).tolist() == [3.0]
    assert cp.asnumpy(restored_surface["t2"]).tolist() == [13.0]
    assert cp.asnumpy(restored_gf.values).tolist() == [14.0]
    assert snapshots.receipt()["device_bytes"] == before["device_bytes"]
    assert snapshots.receipt()["rollbacks"] == 1
    with pytest.raises(RuntimeError, match="lost committed boundary"):
        snapshots.restore(0)


def test_two_deep_snapshot_preserves_every_mutable_carrier_and_reuses_buffers():
    exercise_two_boundaries(CPUArrays)


@pytest.mark.gpu
def test_cuda_two_deep_snapshot_stream_order_and_rollback():
    import cupy as cp
    with cp.cuda.Stream(non_blocking=True):
        exercise_two_boundaries(cp)


def test_two_deep_never_falls_back_to_host_or_one_slot():
    class Exhausted(CPUArrays):
        @staticmethod
        def array(*args, **kwargs):
            raise MemoryError("card exhausted")
    backend = Backend(CPUArrays)
    snapshots = TwoDeepForecastSnapshotV841({"backend": backend,
        "driver": SimpleNamespace(atmosphere=Atmosphere(Carrier(np.ones(1)), Carrier(np.ones(1))))}, Exhausted)
    with pytest.raises(RuntimeError, match="one-deep would overwrite") as error:
        snapshots.capture(0, None, None)
    assert isinstance(error.value.__cause__, MemoryError)


def test_snapshot_restores_regional_clock_counters_flags_and_observer_metadata():
    @dataclass
    class Offsets:
        dynamics_substep: int

    backend = Backend(CPUArrays)
    driving = SimpleNamespace(_state={"u": np.array([3.0])}, _tend={"u": np.array([4.0])},
        _state_scalars=np.array([5.0]), _tend_scalars=np.array([6.0]),
        _pool=SimpleNamespace(_state=Carrier(np.array([7.0])), _interval_start=3))
    discipline = SimpleNamespace(scrubs=11, fused_launches=12, dirty_values=13,
                                  last_kernel="prior", _unit_pool={})
    regional = SimpleNamespace(clock=3, _started=True, _offsets=Offsets(2),
        acoustic_invalid=np.array([0], dtype=np.int32), driving=driving, discipline=discipline)
    driver = SimpleNamespace(atmosphere=Atmosphere(Carrier(np.array([1.0])), Carrier(np.array([2.0]))),
                             regional_v841=regional, mixing_calls_v841=14)
    graph = SimpleNamespace(stats={"replayed": 15, "nested": {"value": 16}}, graph=None)
    park = SimpleNamespace(_receipt={"parks": 17}, _seam=backend._seam)
    snapshot = TwoDeepForecastSnapshotV841({"driver": driver, "backend": backend}, CPUArrays,
        observers={"graphs": graph, "physics_park": park})
    snapshot.capture(3, {}, None)
    regional.clock = 99
    regional._started = False
    regional._offsets.dynamics_substep = 99
    regional.acoustic_invalid.fill(1)
    driving._state["u"].fill(99)
    driving._tend["u"].fill(99)
    driving._state_scalars.fill(99)
    driving._tend_scalars.fill(99)
    driving._pool._state = Carrier(np.array([99.0]))
    driving._pool._interval_start = 99
    discipline.scrubs = discipline.fused_launches = discipline.dirty_values = 99
    discipline.last_kernel = "future"
    driver.mixing_calls_v841 = 99
    graph.stats["nested"]["value"] = 99
    graph.graph = object()
    park._receipt["parks"] = 99
    snapshot.restore(3)
    assert regional.clock == 3 and regional._started
    assert regional._offsets.dynamics_substep == 2 and regional.acoustic_invalid.tolist() == [0]
    assert driving._state["u"].tolist() == [3.0] and driving._tend["u"].tolist() == [4.0]
    assert driving._state_scalars.tolist() == [5.0] and driving._tend_scalars.tolist() == [6.0]
    assert driving._pool._state.values.tolist() == [7.0] and driving._pool._interval_start == 3
    assert (discipline.scrubs, discipline.fused_launches, discipline.dirty_values) == (11, 12, 13)
    assert discipline.last_kernel == "prior" and driver.mixing_calls_v841 == 14
    assert graph.stats == {"replayed": 15, "nested": {"value": 16}} and graph.graph is None
    assert park._receipt == {"parks": 17} and park._seam is backend._seam


def receipt_result():
    return SimpleNamespace(committed=SimpleNamespace(receipt=Carrier(7)),
                           backend_receipt={"phase": "complete"},
                           clamp_d2h=SimpleNamespace(as_dict=lambda: {"bytes": 4}),
                           recovery=SimpleNamespace(receipt=lambda: {"ok": True}))


def test_receipt_serialization_runs_in_background_in_order(tmp_path):
    result = receipt_result()
    launch_thread = threading.get_ident()
    result.recovery.receipt = lambda: {"writer_thread": threading.get_ident()}
    path = tmp_path / "steps.jsonl"
    writer = StepReceiptWriter(path)
    for step in range(1, 20):
        writer.submit(step, 0.5, result, {"ok": True})
    rows = writer.close()
    assert [row["step"] for row in rows] == list(range(1, 20))
    assert all(row["recovery"]["writer_thread"] != launch_thread for row in rows)
    assert [json.loads(line) for line in path.read_text().splitlines()] == rows
    assert writer.close() == rows
    with pytest.raises(RuntimeError, match="closed"):
        writer.submit(20, 0.5, result, {})


def test_writer_failure_is_reported_after_join(tmp_path):
    writer = StepReceiptWriter(tmp_path / "steps.jsonl")
    result = receipt_result()
    result.recovery.receipt = lambda: {"unserializable": object()}
    writer.submit(1, 0.5, result, {})
    with pytest.raises(TypeError):
        writer.close()
    assert writer.closed and writer.stream.closed


class _LoopDone(Exception):
    pass


def _drive_forecast_loop(monkeypatch, tmp_path, refuse_step, *, stop_on_refusal):
    """Run the actual forecast loop with host actors, not a second loop.

    Returns what the loop left behind; the injected step-receipt write ends
    the run (``_LoopDone``), and any other exception propagates as raised.
    """
    import sys
    from woof.hex.drivers import run_cuda_v841_forecast as forecast
    from woof.hex import cuda_backend, cuda_column_order_v841, cuda_driver_lts
    events, written = [], {}
    backend = Backend(CPUArrays)
    driver = SimpleNamespace(atmosphere=Atmosphere(Carrier(np.ones(2)), Carrier(np.ones(1))),
                             cache=object(), config=SimpleNamespace(config_dynamics_split_steps=1))
    stack = {"driver": driver, "backend": backend, "physics_geometry": None}
    monkeypatch.setitem(sys.modules, "cupy", CPUArrays)
    monkeypatch.setattr(forecast, "_pin_for_run", lambda *args: {})
    capability = SimpleNamespace(compute=(12, 0), as_dict=lambda: {})
    monkeypatch.setattr(cuda_backend, "require_cuda", lambda **kwargs: capability)
    monkeypatch.setattr(cuda_backend, "KernelCache", lambda **kwargs: object())
    monkeypatch.setattr(forecast.proof, "gpu_memory_admission", lambda *args, **kwargs: {})
    monkeypatch.setattr(forecast.proof, "_construct_device_stack", lambda **kwargs: stack)
    monkeypatch.setattr(cuda_driver_lts, "attach_local_timestep", lambda *args, **kwargs: None)
    # The column-order coupler reads the prepared mesh, which these host
    # actors do not build; the loop under test never touches its result.
    monkeypatch.setattr(cuda_column_order_v841, "attach_column_order", lambda *args, **kwargs: None)
    monkeypatch.setattr(forecast.proof, "_static_output_fields", lambda host: {})
    monkeypatch.setattr(forecast.proof, "_previous_surface_updates", lambda *args: {})
    monkeypatch.setattr(forecast.history_writer, "open_writer", lambda *args: None)
    monkeypatch.setattr(forecast.proof, "require_arwen_v2_surface_execution", lambda *args, **kwargs: {})
    monkeypatch.setattr(forecast.time_ledger, "emit", lambda stage, phase="mark", **kwargs:
                        events.append(("ledger", stage, phase, kwargs.get("attrs", {}))))
    monkeypatch.setenv("WOOF_HEX_CUDA_GRAPHS", "0")

    def install(stack, cp):
        def numeric(atmosphere, flag):
            step = int(atmosphere.state.time_seconds / forecast.DT_SECONDS)
            events.append(("wait", step))
            driver._step_gate_envelope = (atmosphere, {"step": step})
            return 0
        driver.final_validation_reader = numeric

    monkeypatch.setattr(forecast, "install_step_gate_batch", install)

    def execute(**kwargs):
        step = int(driver.atmosphere.state.time_seconds / forecast.DT_SECONDS) + 1
        events.append(("launch", step))
        endpoint = Atmosphere(Carrier(np.full(2, step), step * forecast.DT_SECONDS),
                              Carrier(np.full(1, step)))
        driver.final_validation_reader(endpoint, None)
        driver.atmosphere = endpoint
        backend._seam.values.fill(step)
        backend._seam.step = step
        backend._gwdo_calls = step
        backend._last_receipt = {"step": step}
        events.append(("publish", step))
        result = receipt_result()
        result.committed = SimpleNamespace(receipt=Carrier(step), surface_updates={"value": np.full(1, step)})
        result.dynamics_tendencies = Carrier(np.full(1, step))
        return result

    def health(stack, step, cp, **kwargs):
        events.append(("health", step))
        assert kwargs["measured_health"][1]["step"] == step
        if step == refuse_step:
            raise FloatingPointError(f"injected health refusal at step {step}")
        return {"step": step}

    monkeypatch.setattr(forecast.proof, "execute_composite_step", execute)
    monkeypatch.setattr(forecast, "step_health_gate", health)

    def write(path, value):
        written[path.name] = value
        if path.name == "snapshot-hashes.json":
            written["step-receipts.json"] = json.loads((path.parent / "step-receipts.json").read_text())
            raise _LoopDone

    monkeypatch.setattr(forecast.proof, "_write_exclusive_json", write)
    out = SimpleNamespace(events=events, written=written, driver=driver, backend=backend,
                          forecast=forecast)
    try:
        forecast.execute_forecast(host={"scalar_names": ("qv",), "surface_accumulators": (),
                                        "prepared": SimpleNamespace(mesh=None)},
            schedule={"capture_steps": [], "labels": {}, "steps": 4}, cache_root=tmp_path,
            output_root=tmp_path, arwen_checkout=tmp_path, source_receipt={}, authority_receipt={},
            fingerprint_every=0, stop_on_refusal=stop_on_refusal)
    except _LoopDone:
        return out
    raise AssertionError("the loop returned before writing its step receipts")


@pytest.mark.parametrize("refuse_step", [0, 1, 2, 4])
def test_main_loop_inspects_prior_health_before_successor_wait_and_truncates(monkeypatch, tmp_path, refuse_step):
    """Exercise the actual loop's ordering with host actors, not a second loop."""
    out = _drive_forecast_loop(monkeypatch, tmp_path, refuse_step, stop_on_refusal=True)
    events, written, driver, backend, forecast = (
        out.events, out.written, out.driver, out.backend, out.forecast)
    endpoint = refuse_step or 4
    assert driver.atmosphere.state.time_seconds == endpoint * forecast.DT_SECONDS
    assert backend._seam.step == endpoint and backend._gwdo_calls == endpoint
    assert [row["step"] for row in written["step-receipts.json"]["receipts"]] == list(range(1, endpoint + 1))
    assert [row[1] for row in events if row[0] == "health"] == list(range(1, endpoint + 1))
    if endpoint < 4:
        assert events.index(("launch", endpoint + 1)) < events.index(("health", endpoint)) < events.index(("wait", endpoint + 1))
        rollback_marks = [row for row in events if row[:3] == ("ledger", "forecast.integrate", "mark")]
        assert rollback_marks[0][3]["event"] == "rollback"


@pytest.mark.parametrize("refuse_step", [0, 2])
def test_run_without_stop_on_refusal_retains_no_boundary(monkeypatch, tmp_path, refuse_step):
    """Without --stop-on-refusal nothing reads a retained boundary, so none is copied.

    THE BREAKAGE THIS PREVENTS (measured 2026-10-11, one 96 GB card, 30 km
    x1.655362, bare defaults): the two-deep late-health snapshot held
    21,540 MiB of device copies that no reader ever restored, and the run ran
    out of card memory at its second step.
    """
    from woof.hex import cuda_seam_snapshot_v841

    def refuse_construction(*args, **kwargs):
        raise AssertionError("a run without --stop-on-refusal built the two-deep snapshot")

    monkeypatch.setattr(cuda_seam_snapshot_v841, "TwoDeepForecastSnapshotV841",
                        refuse_construction)
    if refuse_step:
        # A late health refusal ends the run with the refusal itself; no
        # rollback is attempted and no successor is restored.
        with pytest.raises(FloatingPointError, match=f"refusal at step {refuse_step}"):
            _drive_forecast_loop(monkeypatch, tmp_path, refuse_step, stop_on_refusal=False)
        return
    out = _drive_forecast_loop(monkeypatch, tmp_path, refuse_step, stop_on_refusal=False)
    assert [row["step"] for row in out.written["step-receipts.json"]["receipts"]] == [1, 2, 3, 4]
    assert not [row for row in out.events if row[:3] == ("ledger", "forecast.integrate", "mark")]


def test_unarmed_snapshot_holds_nothing_and_refuses_a_restore():
    from woof.hex.drivers.run_cuda_v841_forecast import UnarmedForecastSnapshotV841

    unarmed = UnarmedForecastSnapshotV841()
    unarmed.capture(0, {"t2": np.ones(3)}, None)
    receipt = unarmed.receipt()
    assert receipt["depth"] == 0 and receipt["device_bytes"] == 0 and receipt["captures"] == 0
    with pytest.raises(RuntimeError, match="retained no boundary"):
        unarmed.restore(0)


@pytest.mark.parametrize("steps", [0, 1, 19])
def test_streamed_compatibility_document_keeps_exact_bytes(tmp_path, steps):
    path, document = tmp_path / "steps.jsonl", tmp_path / "steps.json"
    writer = StepReceiptWriter(path, document=document, schema="test/steps")
    for step in range(steps):
        writer.submit(step, 0.5, receipt_result(), {"unicode": "\u00e9", "empty": []})
    assert not document.exists()
    rows = writer.close()
    expected = json.dumps({"schema": "test/steps", "receipts": rows},
                          indent=2, sort_keys=True, allow_nan=False) + "\n"
    assert document.read_bytes() == expected.encode("utf-8")
    assert writer.close() == rows


def test_streamed_document_does_not_replace_existing_receipt(tmp_path):
    document = tmp_path / "steps.json"
    document.write_text("original")
    writer = StepReceiptWriter(tmp_path / "steps.jsonl", document=document, schema="test")
    with pytest.raises(FileExistsError):
        writer.close()
    assert document.read_text() == "original"
    assert not list(tmp_path.glob(".*.tmp"))
