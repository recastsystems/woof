"""The host thread waits for the card with blocking sync, not CUDA's spin (woof/cuda_wait.py).

Breakage it prevents: a GPU-bound WOOF-ArWen forecast holding one host core at 100 % for its whole step
loop (a 3 km GFS nest forecast, 2026-10-27: perf put the main thread 67 % in libcuda, 16 % in sched_yield).
Driven with a fake driver: no card needed.
"""

import ctypes

import pytest

from woof import cuda_wait


class FakeDriver:
    """Primary contexts per ordinal: {ordinal: [flags, active]}; v2 entry points only."""

    def __init__(self, devices, *, has_set=True, init_rc=0):
        self.devices = {k: list(v) for k, v in devices.items()}
        self.init_rc = init_rc
        self.current = []
        self.retained = {k: 0 for k in devices}
        self.live = {k: v[0] for k, v in devices.items()}
        self.created = []
        if has_set:
            self.cuCtxSetFlags = self._set

    def cuInit(self, flags):
        return self.init_rc

    def cuDeviceGetCount(self, out):
        ctypes.cast(out, ctypes.POINTER(ctypes.c_int))[0] = len(self.devices)
        return 0

    def cuDeviceGet(self, out, ordinal):
        ctypes.cast(out, ctypes.POINTER(ctypes.c_int))[0] = ordinal
        return 0

    def cuDevicePrimaryCtxGetState(self, dev, flags, active):
        f, a = self.devices[dev.value if hasattr(dev, "value") else dev]
        ctypes.cast(flags, ctypes.POINTER(ctypes.c_uint))[0] = f
        ctypes.cast(active, ctypes.POINTER(ctypes.c_int))[0] = int(a)
        return 0

    def cuDevicePrimaryCtxSetFlags_v2(self, dev, flags):
        ordinal = dev.value if hasattr(dev, "value") else dev
        self.devices[ordinal][0] = flags.value
        return 0

    def cuDevicePrimaryCtxRetain(self, out, dev):
        ordinal = dev.value if hasattr(dev, "value") else dev
        if not self.devices[ordinal][1]:
            self.created.append(ordinal)
        self.retained[ordinal] += 1
        ctypes.cast(out, ctypes.POINTER(ctypes.c_void_p))[0] = 0x1000 + ordinal
        return 0

    def cuDevicePrimaryCtxRelease_v2(self, dev):
        self.retained[dev.value if hasattr(dev, "value") else dev] -= 1
        return 0

    def cuCtxPushCurrent_v2(self, ctx):
        self.current.append(ctx.value - 0x1000)
        return 0

    def cuCtxPopCurrent_v2(self, out):
        self.current.pop()
        return 0

    def cuCtxGetFlags(self, out):
        ctypes.cast(out, ctypes.POINTER(ctypes.c_uint))[0] = self.live[self.current[-1]]
        return 0

    def _set(self, flags):
        self.live[self.current[-1]] = flags.value
        return 0


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(cuda_wait, "_applied", {})
    monkeypatch.setattr(cuda_wait, "_warned", False)


def test_active_context_gets_blocking_sync_and_keeps_its_other_flags():
    driver = FakeDriver({0: (0x08, True)})  # LMEM_RESIZE_TO_MAX, auto scheduling
    assert cuda_wait.apply(environ={}, driver=driver) == {0: "blocking active (context)"}
    assert driver.live[0] == 0x08 | 0x04
    assert driver.retained[0] == 0 and driver.current == []  # balanced


def test_an_inactive_card_gets_creation_flags_and_no_context():
    driver = FakeDriver({0: (0x08, True), 1: (0x00, False)})
    result = cuda_wait.apply(environ={}, driver=driver)
    assert result[1] == "blocking inactive (creation flags)"
    assert driver.devices[1][0] == 0x04
    assert driver.created == []  # never stands a context up


def test_already_blocking_is_left_alone():
    driver = FakeDriver({0: (0x04, True)})
    driver.live[0] = 0x04
    assert cuda_wait.apply(environ={}, driver=driver) == {0: "blocking active (already)"}


def test_old_driver_without_ctx_set_flags_says_so_once(capsys):
    driver = FakeDriver({0: (0x00, True)}, has_set=False)
    assert cuda_wait.apply(environ={}, driver=driver) == {0: "unchanged"}
    assert "WOOF_CUDA_SCHEDULE" in capsys.readouterr().err
    cuda_wait.apply(environ={}, driver=driver)
    assert capsys.readouterr().err == ""


def test_no_visible_card_or_no_driver_is_a_no_op():
    assert cuda_wait.apply(environ={}, driver=FakeDriver({0: (0, True)}, init_rc=100)) == {}
    assert cuda_wait.apply(environ={"WOOF_CUDA_SCHEDULE": "auto"}, driver=object()) == {}


def test_knob():
    driver = FakeDriver({0: (0x04, True)})
    driver.live[0] = 0x04
    assert cuda_wait.apply(environ={"WOOF_CUDA_SCHEDULE": "spin"}, driver=driver) == {
        0: "spin active (context)"}
    assert driver.live[0] == 0x01
    with pytest.raises(ValueError, match="WOOF_CUDA_SCHEDULE"):
        cuda_wait.requested({"WOOF_CUDA_SCHEDULE": "fast"})


def test_the_step_loop_and_the_cuda_preparation_both_ask_for_it():
    import inspect

    from woof.core import model
    from woof.ingest import preprocess_backend

    assert "_apply_cuda_wait()" in inspect.getsource(model.execute_experiment)
    source = inspect.getsource(preprocess_backend._resolve_preprocess_backend)
    assert source.count("_wait_for_the_card_blocking()") == 2


def _smi(stdout, returncode=0):
    import subprocess

    def run(*args, **kwargs):
        return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr="")
    return run


def test_a_multi_card_preparation_with_no_mask_takes_the_least_busy_card(monkeypatch, capsys):
    """Breakage: every unmasked GFS preparation on a rented 8-card box held a context on card 0, another lane's."""
    from woof.ingest import preprocess_backend as pb

    monkeypatch.setattr(pb, "_cuda_not_yet_initialized", lambda: True)
    monkeypatch.setattr(pb.subprocess, "run", _smi(
        "0, GPU-a, 2000, 100\n1, GPU-b, 30000, 0\n2, GPU-c, 31000, 80\n3, GPU-d, 29000, 10\n"))
    environ = {}
    assert pb._mask_to_the_least_busy_card(environ) == "GPU-b"
    assert environ == {"CUDA_VISIBLE_DEVICES": "GPU-b"}
    assert "card 1" in capsys.readouterr().err


@pytest.mark.parametrize("environ,stdout,initialized", [
    ({"CUDA_VISIBLE_DEVICES": "3"}, "0, GPU-a, 1, 100\n1, GPU-b, 9, 0\n", False),  # a mask is the caller's choice
    ({}, "0, GPU-a, 9000, 0\n", False),                                          # one card: nothing to choose
    ({}, "0, GPU-a, 1, 90\n1, GPU-b, 9, 70\n", False),                             # all busy: auto reads it as before
    ({}, "0, GPU-a, 1, 0\n1, GPU-b, 9, 0\n", True),                               # CUDA already started: too late
])
def test_the_card_choice_stays_out_of_the_way(monkeypatch, environ, stdout, initialized):
    from woof.ingest import preprocess_backend as pb

    monkeypatch.setattr(pb, "_cuda_not_yet_initialized", lambda: not initialized)
    monkeypatch.setattr(pb.subprocess, "run", _smi(stdout))
    before = dict(environ)
    assert pb._mask_to_the_least_busy_card(environ) is None
    assert environ == before
