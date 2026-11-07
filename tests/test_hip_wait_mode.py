from woof import hip_runtime


class Hip:
    def __init__(self, active):
        self.active = active
        self.creation_flags = None
        self.error = 0

    def hipPeekAtLastError(self):
        return self.error

    def hipGetLastError(self):
        error, self.error = self.error, 0
        return error

    def hipGetDevice(self, pointer):
        pointer._obj.value = 0
        return 0

    def hipDevicePrimaryCtxGetState(self, ordinal, flags, active):
        flags._obj.value = 8
        active._obj.value = self.active
        return 0

    def hipDevicePrimaryCtxSetFlags(self, ordinal, flags):
        self.creation_flags = flags
        return 0

    def hipSetDeviceFlags(self, flags):
        raise AssertionError('must not poison the live HIP last-error slot')


def test_active_context_does_not_attempt_unsupported_flag_change(monkeypatch):
    hip = Hip(active=1)
    monkeypatch.setattr(hip_runtime, 'library', lambda: hip)
    assert hip_runtime.wait_mode('blocking', [0]) == {
        0: 'blocking unchanged (active HIP context)'}
    assert hip.creation_flags is None


def test_inactive_context_preserves_non_scheduling_flags(monkeypatch):
    hip = Hip(active=0)
    monkeypatch.setattr(hip_runtime, 'library', lambda: hip)
    assert hip_runtime.wait_mode('yield', [0]) == {0: 'yield creation flags'}
    assert hip.creation_flags == 10


def test_rocm_rejected_creation_flags_do_not_poison_thrust(monkeypatch):
    hip = Hip(active=0)
    def reject(*args):
        hip.error = 216
        return 216
    hip.hipDevicePrimaryCtxSetFlags = reject
    monkeypatch.setattr(hip_runtime, 'library', lambda: hip)
    assert hip_runtime.wait_mode('blocking', [0]) == {0: 'blocking unchanged (HIP 216)'}
    assert hip.error == 0


def test_preexisting_gpu_error_is_preserved(monkeypatch):
    hip = Hip(active=0)
    hip.error = 700
    monkeypatch.setattr(hip_runtime, 'library', lambda: hip)
    assert hip_runtime.wait_mode('blocking', [0]) == {
        0: 'blocking unchanged (pre-existing HIP 700)'}
    assert hip.error == 700
    assert hip.creation_flags is None
