"""HIP-on-NVIDIA certificates bind NVCC, without loading HIPRTC."""
import hashlib
from woof import hip_backend


def test_nvidia_aot_identity_does_not_load_hiprtc(monkeypatch, tmp_path):
    from woof import hip_backend, hiprtc
    compiler = tmp_path / 'bin' / 'nvcc'
    compiler.parent.mkdir()
    compiler.write_bytes(b'actual-aot-compiler')
    monkeypatch.setenv('CUDA_PATH', str(tmp_path))
    monkeypatch.setenv('HIPCC_NVIDIA', '/private/hipcc-nvidia')
    monkeypatch.setattr(hip_backend, 'requested', lambda: 'hip-nvidia')
    monkeypatch.setattr(hip_backend, 'toolchain_identity', lambda: ('/hip/hipcc', 'HIP'))
    monkeypatch.setattr(hip_backend, 'nvidia_banner', lambda path: 'NVCC 12.9')
    def unused(): raise AssertionError('this mode never compiles with HIPRTC')
    monkeypatch.setattr(hiprtc, 'identity', unused)
    result = hip_backend.compiler_runtime_identity()
    assert result['mode'] == 'hipcc-nvidia-aot'
    assert result['compiler'] == str(compiler)
    assert result['library_sha256']


def test_matched_runtime_compiler_binds_nvrtc_without_nvcc(monkeypatch):
    from woof import hip_backend, hip_nvidia_rtc
    monkeypatch.setenv('WOOF_HIP_NVIDIA_COMPILER', 'nvrtc')
    monkeypatch.setattr(hip_backend, 'requested', lambda: 'hip-nvidia')
    def unused(): raise AssertionError('runtime compiler does not use NVCC')
    monkeypatch.setattr(hip_backend, 'toolchain_identity', unused)
    expected = dict(mode='hiprtc-nvidia', version=[13, 2], library_sha256='real-nvrtc-hash')
    monkeypatch.setattr(hip_nvidia_rtc, 'identity', lambda: expected)
    assert hip_backend.compiler_runtime_identity() == expected


def test_amd_route_still_measures_hiprtc(monkeypatch):
    from woof import hiprtc
    monkeypatch.setenv('WOOF_BACKEND', 'amd')
    measured = {'version': [7, 2], 'library_sha256': 'actual-hiprtc-hash'}
    monkeypatch.setattr(hiprtc, 'identity', lambda: measured)
    assert hip_backend.runtime_compiler_identity() == measured


def test_nvidia_identity_binds_implicit_math_inputs(tmp_path, monkeypatch):
    compiler = tmp_path/'bin/nvcc'
    library = tmp_path/'nvvm/libdevice/libdevice.10.bc'
    for path, data in ((compiler, b'compiler'), (library, b'math inputs')):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    monkeypatch.setattr(hip_backend, 'nvidia_banner', lambda compiler: 'test compiler')
    hip_backend.nvidia_compiler_identity.cache_clear()
    first = hip_backend.nvidia_compiler_identity(str(compiler))
    library.write_bytes(b'different math inputs')
    hip_backend.nvidia_compiler_identity.cache_clear()
    second = hip_backend.nvidia_compiler_identity(str(compiler))
    assert first['library_sha256'] != second['library_sha256']
    assert first['compiler'] == str(compiler)
