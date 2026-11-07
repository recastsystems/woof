"""A translated MYNN image must never replace the CUDA cache entry."""
import base64
import lzma
from pathlib import Path
from types import SimpleNamespace
import sys

from woof import hip_backend
from woof.hex import mynn_glibc_math
from woof.hex.cuda_backend import aot, aot_build
from woof.hex.mpasphys import resources_build


def test_cuda_and_hip_resource_images_are_separate(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(__file__=str(tmp_path/'cupy.py')))
    monkeypatch.setattr(mynn_glibc_math, 'mynn_glibc_source', lambda: 'fixture maths')
    state = dict(hip=False)
    monkeypatch.setattr(hip_backend, 'active', lambda: state['hip'])
    monkeypatch.setattr(hip_backend, 'compiler_binding', lambda: {'compiler': 'matched NVRTC'})
    class Pack:
        def select(self, *args, **kwargs):
            return None, 'fixture'
    monkeypatch.setattr(aot, 'Pack', Pack)
    compiled = []
    def compile(row, arch):
        image = b'HIP image' if state['hip'] else b'CUDA image'
        compiled.append(image)
        return {'cubin_xz_base64': base64.b64encode(lzma.compress(image)).decode()}
    monkeypatch.setattr(aot_build, 'compile_entry', compile)
    paths = []
    for hip in (False, True, False, True):
        state['hip'] = hip
        result = resources_build.build_mynnedmf_image(tmp_path, arch='sm_120', log=lambda _: None)
        paths.append(result['image'])
        assert Path(result['image']).read_bytes() == (b'HIP image' if hip else b'CUDA image')
    assert paths[0] != paths[1]
    assert paths[:2] == paths[2:]
    assert compiled == [b'CUDA image', b'HIP image']
