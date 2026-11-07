"""AMD compiler invalidation must retain the release CUDA disk cache."""
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('scheme', ['mynn', 'gf'])
def test_existing_release_cuda_image_is_reused(tmp_path, monkeypatch, scheme):
    from woof import hip_backend
    from woof.hex.cuda_backend import aot
    from woof.hex.mpasphys import bl_mynn_v842 as mynn, gf_mpas_v842 as gf
    loaded = []

    class Device:
        compute_capability = '90'
        def __init__(self, *args): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

    class Module:
        def load(self, image): loaded.append(image)

    cp = SimpleNamespace(__version__='cache-regression', cuda=SimpleNamespace(
        Device=Device, function=SimpleNamespace(Module=Module)))
    nvrtc = SimpleNamespace(getVersion=lambda: (12, 9))
    monkeypatch.setitem(sys.modules, 'cupy', cp)
    monkeypatch.setitem(sys.modules, 'cupy.cuda', SimpleNamespace(compiler=object()))
    monkeypatch.setitem(sys.modules, 'cupy_backends.cuda.libs', SimpleNamespace(nvrtc=nvrtc))
    monkeypatch.setenv('CUPY_CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(hip_backend, 'active', lambda: False)
    monkeypatch.setattr(mynn, '_compilation_target', lambda **kw: '90')
    monkeypatch.setattr(gf, '_compilation_target', lambda *a, **kw: '90')
    monkeypatch.setattr(mynn._hex_numerics, 'options', lambda options: tuple(options))

    def unexpected(*args, **kwargs):
        raise AssertionError('A release CUDA cache hit must not probe a new compiler identity or recompile')

    monkeypatch.setattr(aot, 'compiler_identity', unexpected)
    monkeypatch.setattr(aot, 'direct_image', unexpected)
    monkeypatch.setattr(aot, 'raw_image', unexpected)
    source='extern "C" __global__ void cached() {}'
    identity=dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  nvrtc=[12,9], cupy=cp.__version__)
    if scheme == 'mynn':
        identity.update(options=list(mynn.MYNN_CUDA_OPTIONS), arch='90')
        directory=tmp_path/'mpas_mynn_v842'; suffix='.bin'
    else:
        identity.update(filename='gf_mpas_v842.cu',
                        options=[*gf.COMPILER_OPTIONS, '--gpu-architecture=sm_90'])
        directory=tmp_path/'hex-gf-mpas'; suffix='.cubin'
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    directory.mkdir()
    image=b'unchanged release CUDA image'
    (directory/(key+suffix)).write_bytes(image)
    (directory/(key+'.json')).write_text(json.dumps(dict(identity=identity,
        binary_sha256=hashlib.sha256(image).hexdigest())))
    if scheme == 'mynn': mynn._compile_device_module(source)
    else: gf._compile_module(source,0)
    assert loaded == [image]


@pytest.mark.parametrize('revision', ['c6d08709', 'c942418'])
def test_rrfs_cuda_keeps_the_release_direct_nvrtc_route(monkeypatch, revision):
    import importlib
    from woof import hip_backend
    from woof.hex.cuda_backend import aot
    from woof.hex import cuda_fp32
    module = importlib.import_module('woof.hex.mpasphys.rrfs.bl_mynnedmf_'+revision)
    calls=[]
    class Image:
        def load(self, value): calls.append(('load',value))
    cp=SimpleNamespace(cuda=SimpleNamespace(
        Device=lambda: SimpleNamespace(compute_capability='90'),
        function=SimpleNamespace(Module=Image)))
    nvrtc=SimpleNamespace(
        createProgram=lambda *args: calls.append(('create',args)) or 7,
        compileProgram=lambda *args: calls.append(('compile',args)),
        getCUBIN=lambda *args: calls.append(('image',args)) or b'release cubin',
        destroyProgram=lambda *args: calls.append(('destroy',args)))
    monkeypatch.setitem(sys.modules,'cupy',cp)
    monkeypatch.setitem(sys.modules,'cupy_backends.cuda.libs',SimpleNamespace(nvrtc=nvrtc))
    monkeypatch.setattr(hip_backend,'active',lambda:False)
    monkeypatch.setattr(module._hex_numerics,'options',lambda options:tuple(options))
    monkeypatch.setattr(cuda_fp32,'CUDA_FTZ_HELPERS','/* helpers */')
    def unexpected(*args,**kwargs):
        raise AssertionError('CUDA must keep the release direct NVRTC route')
    monkeypatch.setattr(aot,'raw_image',unexpected)
    monkeypatch.setattr(module.MynnEdmfCuda,'_load_kernels',lambda self:None)
    backend=module.MynnEdmfCuda(shared_math_source='/* math */')
    backend._compile()
    source='/* helpers *//* math */'+module.ARRAY_SUPPORT+module.MODULE_CUDA
    options=tuple(module.CUDA_OPTIONS)+('--gpu-architecture=sm_90',)
    assert calls==[('create',(source,'rrfs_mynnedmf.cu',(),())),
                   ('compile',(7,options)),('image',(7,)),('destroy',(7,)),
                   ('load',b'release cubin')]
