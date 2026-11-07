"""HIP-only host ABI checks; no card or CUDA library required."""
import ctypes as C
from types import SimpleNamespace


def test_wait_configuration_preserves_flags_and_reports_refusal(monkeypatch):
    from woof import hip_runtime
    flags_seen = []
    class Hip:
        def hipGetDevice(self, p): p._obj.value = 0; return 0
        def hipDevicePrimaryCtxGetState(self, d, f, a):
            f._obj.value = 8; a._obj.value = int(d == 0); return 0
        def hipDevicePrimaryCtxSetFlags(self, d, flags): flags_seen.append((d, flags)); return 0
        def hipPeekAtLastError(self): return 0
        def hipSetDeviceFlags(self, flags): raise AssertionError('active context is immutable')
    monkeypatch.setattr(hip_runtime, 'library', lambda: Hip())
    result = hip_runtime.wait_mode('blocking', [0, 1])
    assert flags_seen == [(1, 12)]
    assert result == {0: 'blocking unchanged (active HIP context)', 1: 'blocking creation flags'}


def test_hip_stack_identity_keeps_gfx(monkeypatch):
    from woof.gpu_stack_identity import _hip_stack_identity
    cp = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(driverGetVersion=lambda: 60400000)))
    row = _hip_stack_identity(cp, {'gcnArchName': b'gfx942'}, 'card', 1, 60400000, 120.)
    assert row['backend'] == 'hip' and row['architecture'] == 'gfx942'


def test_rocm_preprocessing_does_not_decode_hip_version_as_cuda():
    from woof.ingest.preprocess_backend import _certified_cuda
    # HIP's version integer has a different encoding. Do not read it through
    # the CUDA-major certification table or claim a CUDA 70253 runtime.
    cp = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(is_hip=True)))
    certified, reason = _certified_cuda(cp)
    assert not certified
    assert 'ROCm/HIP' in reason and 'table does not apply' in reason


def test_hip_warm_image_binds_compiler_config_paths_without_rejecting_them(tmp_path, monkeypatch):
    from woof import hip_backend as hip
    from woof.kernel_warm_pack import WarmImages
    root = tmp_path / 'rocm'
    (root / 'include').mkdir(parents=True)
    (root / 'include/config.h').write_text('#define COMPILER "/opt/rocm/bin/clang"\n')
    external = tmp_path / 'cupy'
    external.mkdir()
    (external / 'array.h').write_text('struct array {};\n')
    identity = ['compiler-v1']
    monkeypatch.setattr(hip, 'active', lambda: True)
    monkeypatch.setattr(hip, 'toolchain_identity', lambda: (str(root / 'bin/hipcc'), 'test'))
    monkeypatch.setattr(hip, 'compiler_dependencies', lambda _: {'sha': identity[0]})
    arguments = dict(source='kernel', options=('-I'+str(root/'include'), '-I'+str(external)))
    first = WarmImages.key(arguments)
    identity[0] = 'compiler-v2'
    assert WarmImages.key(arguments) != first


def test_hip_warm_image_still_rejects_external_header_escapes(tmp_path, monkeypatch):
    import pytest
    from woof import hip_backend as hip
    from woof.kernel_warm_pack import WarmImages
    external = tmp_path / 'external'
    external.mkdir()
    (external / 'array.h').write_text('#include "/outside/unknown.h"\n')
    monkeypatch.setattr(hip, 'active', lambda: True)
    monkeypatch.setattr(hip, 'toolchain_identity', lambda: (str(tmp_path/'rocm/bin/hipcc'), 'test'))
    monkeypatch.setattr(hip, 'compiler_dependencies', lambda _: {'sha': 'compiler'})
    with pytest.raises(ValueError, match='escape'):
        WarmImages.key(dict(source='kernel', options=('-I'+str(external),)))
