"""An explicitly selected AMD doctor must fail honestly and skip CUDA probes."""
from woof.hex import doctor


def test_failed_amd_smoke_is_a_required_missing_finding(monkeypatch):
    monkeypatch.setenv("WOOF_BACKEND", "amd")
    from woof import hip_doctor
    report=dict(ok=False,backend="amd",rocm_version="7.2",gfx_target="gfx942",wave_size=64,kernel_smoke=False)
    monkeypatch.setattr(hip_doctor,"run",lambda:report)
    findings=doctor.check_gpu_runtime()
    assert doctor.blocking_gaps(findings)==findings
    assert findings[0].evidence==report


def test_amd_architecture_comes_from_the_hip_smoke(monkeypatch):
    monkeypatch.setenv("WOOF_BACKEND", "amd")
    def forbidden():raise AssertionError("AMD doctor must not probe CUDA")
    monkeypatch.setattr(doctor,"_card_probe",forbidden)
    assert doctor.check_gpu_architecture()==[]
