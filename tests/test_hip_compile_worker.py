"""The parallel proof route must use the compiler named by its certificate."""
from woof import hip_backend, hip_nvidia_rtc, kernel_compile_worker


def test_matched_nvidia_worker_does_not_use_aot_or_hiprtc(monkeypatch):
    monkeypatch.setenv('WOOF_HIP_NVIDIA_COMPILER', 'nvrtc')
    monkeypatch.setattr(hip_backend, 'active', lambda: True)
    monkeypatch.setattr(hip_backend, 'requested', lambda: 'hip-nvidia')
    monkeypatch.setattr(hip_backend, 'compiler_binding', lambda: {'compiler': 'nvrtc'})
    def wrong(*args, **kwargs):
        raise AssertionError('parallel matched NVRTC must not compile with NVCC')
    monkeypatch.setattr(hip_backend, 'compile_aot', wrong)
    seen = []
    def compile(arguments, target):
        seen.append((arguments, target))
        return b'matched-image', {'template': 'lowered'}
    monkeypatch.setattr(hip_nvidia_rtc, 'compile_explicit', compile)
    request = dict(source='canonical', options=['-ftz=false'], filename='kernel.cu',
                   name_expressions=['template<float>'])
    result, timing = kernel_compile_worker.compile_request(request, '120', {'compiler': 'nvrtc'})
    assert seen == [(request, 'sm_120')]
    assert result == (b'matched-image', {'template': 'lowered'})
    assert timing['seconds'] >= 0
