"""Auto preparation shares the fit probe and records measured contention."""

from types import SimpleNamespace

import pytest

from woof.core import device_probe, preflight
from woof.ingest import preprocess_backend as backend


@pytest.fixture
def candidates(monkeypatch):
    runtime = SimpleNamespace(getDeviceCount=lambda: 1, getDevice=lambda: 0,
                              runtimeGetVersion=lambda: 13020)
    cuda = SimpleNamespace(name="cuda", array_module=SimpleNamespace(
        __version__="14.2.0", cuda=SimpleNamespace(runtime=runtime)))
    cpu = SimpleNamespace(name="cpu")
    monkeypatch.setattr(backend, "CudaPreprocessBackend", lambda: cuda)
    monkeypatch.setattr(backend, "ParallelCpuPreprocessBackend", lambda **_: cpu)
    monkeypatch.setattr(backend, "_gpu_runtime_installed", lambda: True)
    monkeypatch.setattr(backend, "_ANNOUNCED_AUTO_REASONS", set())
    monkeypatch.delenv("WOOF_NATIVE_DISTRIBUTION_MANIFEST", raising=False)
    # The CUDA stack here is a stand-in, so this machine allows its card:
    # since 083ac7cd0 the backend choice honours GPUWM_NO_LOCAL_GPU first,
    # and the CPU legs set it.
    monkeypatch.delenv("GPUWM_NO_LOCAL_GPU", raising=False)
    return cuda, cpu


@pytest.mark.parametrize(
    "free,utilization,reason_fragment",
    [
        (30, 50, "utilization 50%"),
    ],
    ids=['30-50-utilization 50%'],
)
def test_auto_keeps_cuda_for_measured_busy_card(
        monkeypatch, capsys, candidates, free, utilization, reason_fragment):
    measured = {"free_bytes": free * 2**30, "total_bytes": 32 * 2**30,
                "utilization_gpu_percent": utilization}
    monkeypatch.setattr(device_probe, "device_memory_probe_subprocess", lambda **_: measured)
    chosen = backend.resolve_preprocess_backend("auto")
    assert chosen is candidates[0]
    selection = chosen.selection
    assert "certified" in selection["reason"]
    assert selection["requested"] == "auto"
    load = selection["device_load"]
    assert load["free_bytes"] == measured["free_bytes"]
    assert load["total_bytes"] == measured["total_bytes"]
    assert load["utilization_gpu_percent"] == utilization
    assert load["busy_utilization_threshold_percent"] == 50
    # The free-memory FRACTION is retired (A65): whether a preparation fits
    # is its price against the free memory, weighed by admit_preparation.
    assert "minimum_free_fraction" not in load
    message = capsys.readouterr().err
    assert reason_fragment in message
    assert "keeping CUDA" in message
    assert message.startswith("note:"), "go forwards only named notes/warnings"
    assert message.count("\n") == 1


@pytest.mark.parametrize("probe", [None], ids=['None'])
def test_auto_keeps_idle_or_unmeasured_card_choice(
        monkeypatch, capsys, candidates, probe):
    monkeypatch.setattr(device_probe, "device_memory_probe_subprocess", lambda **_: probe)
    chosen = backend.resolve_preprocess_backend("auto")
    assert chosen is candidates[0]
    assert "certified" in chosen.selection["reason"]
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "error,nvml,fragments",
    [
        ("CUDARuntimeError: cudaErrorDevicesUnavailable: CUDA-capable device(s) "
     "is/are busy or unavailable",
     {"used_bytes": 2**30, "total_bytes": 32 * 2**30,
      "utilization_gpu_percent": 0},
     ("cudaErrorDevicesUnavailable",)),
    ],
    ids=['CUDARuntimeError: cudaErrorDevicesUnavailable: CUDA-capable device(s) is/are busy or unavailable-nvml1-fragments1'],
)
def test_auto_waits_when_a_certified_card_is_temporarily_unavailable(
        monkeypatch, capsys, candidates, error, nvml, fragments):
    import time
    monkeypatch.setattr(time, "sleep", lambda _: None)
    asked = {}
    readings = iter([{"cuda_error": error, "nvml": nvml},
                     {"free_bytes": 30 * 2**30, "total_bytes": 32 * 2**30}])

    def probe(**kwargs):
        asked.update(kwargs)
        return next(readings)

    monkeypatch.setattr(device_probe, "device_memory_probe_subprocess", probe)
    chosen = backend.resolve_preprocess_backend("auto")
    assert chosen is candidates[0], "contention must not change preparation arithmetic"
    assert asked.get("report_failure") is True
    selection = chosen.selection
    message = capsys.readouterr().err
    for fragment in fragments:
        assert fragment in message
    load = selection["initial_device_load"]
    assert load["cuda_error"] == error
    if nvml is None:
        assert load["free_bytes"] is None and load["total_bytes"] is None
        assert load["utilization_gpu_percent"] is None
    else:
        assert load["free_bytes"] == nvml["total_bytes"] - nvml["used_bytes"]
        assert load["total_bytes"] == nvml["total_bytes"]
        assert load["utilization_gpu_percent"] == nvml["utilization_gpu_percent"]
    assert "waiting for" in message
    assert selection["device_wait_seconds"] >= 0


def test_explicit_backend_does_not_apply_auto_load_policy(monkeypatch, candidates):
    def unexpected(**_):
        pytest.fail("an explicit choice queried automatic load policy")
    monkeypatch.setattr(device_probe, "device_memory_probe_subprocess", unexpected)
    assert backend.resolve_preprocess_backend("cuda") is candidates[0]
    assert backend.resolve_preprocess_backend("cpu") is candidates[1]


def test_auto_rechecks_load_for_each_preparation(monkeypatch, candidates):
    values = iter([0, 99])
    monkeypatch.setattr(device_probe, "device_memory_probe_subprocess", lambda **_: {
        "free_bytes": 30 * 2**30, "total_bytes": 32 * 2**30,
        "utilization_gpu_percent": next(values)})
    assert backend.resolve_preprocess_backend("auto") is candidates[0]
    assert backend.resolve_preprocess_backend("auto") is candidates[0]


def test_the_preflight_probe_is_the_one_auto_reads():
    """One probe, two import paths: auto reads the leaf RW-WPS stages, and
    the forecast preflight re-exports that same function to its callers, so
    the fit and the backend choice read one card the same way."""

    for name in ("device_memory_probe_subprocess", "device_memory_probe_reason",
                 "_DEVICE_MEMORY_PROBE_SOURCE", "PROBE_EXIT_NO_RUNTIME",
                 "PROBE_EXIT_CARD_UNREAD", "DEVICE_MEMORY_PROBE_TIMEOUT_SECONDS",
                 "PROBE_REASON_NO_RUNTIME", "_probe_error", "_probe_last_line"):
        assert getattr(preflight, name) is getattr(device_probe, name), name


def test_auto_without_a_device_and_explicit_cpu_pin_still_use_cpu(monkeypatch, candidates):
    runtime = candidates[0].array_module.cuda.runtime
    monkeypatch.setattr(runtime, "getDeviceCount", lambda: 0)
    assert backend.resolve_preprocess_backend("auto") is candidates[1]
    monkeypatch.setattr(runtime, "getDeviceCount", lambda: 1)
    assert backend.resolve_preprocess_backend("cpu") is candidates[1]


@pytest.mark.parametrize("requested", ["auto", "cuda"])
def test_cuda_waits_for_memory_without_changing_arithmetic(monkeypatch, candidates, capsys, requested):
    import time
    from woof.ingest.preparation_price import price_preparation, SourceInventory
    from woof.config import RunConfig
    price = price_preparation("gfs", [RunConfig(nx=50, ny=50, nz=49, dx=12000, dy=12000, ztop=20000, dt=60, run_seconds=120)],
                              SourceInventory(levels=23, level_fields=5, surface_planes=19))
    readings = iter([{"free_bytes": 0, "total_bytes": 32 * 2**30,
                      "utilization_gpu_percent": 99},
                     {"free_bytes": 30 * 2**30, "total_bytes": 32 * 2**30,
                      "utilization_gpu_percent": 99}])
    monkeypatch.setattr(device_probe, "device_memory_probe_subprocess", lambda **_: next(readings))
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))
    chosen = backend.resolve_preprocess_backend(requested, price=price)
    assert chosen is candidates[0]
    assert sleeps and chosen.selection["device_fit"]["fits"] is True
    assert "waiting for" in capsys.readouterr().err
