"""HIP's NVIDIA runtime compiler: translated device source through NVRTC.

HIPRTC on NVIDIA uses NVRTC. Keep this explicit so translation proofs can
use the same compiler build as their CUDA referee, without changing CUDA.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


def identity():
    from cupy_backends.cuda.libs import nvrtc
    from woof.certify.compile_platform import nvrtc_library
    version = list(nvrtc.getVersion())
    library = nvrtc_library()
    return dict(mode='hiprtc-nvidia', version=version,
                library_sha256=library['sha256'])


def compile_explicit(arguments, arch):
    """Compile a plan request without creating a device context in the worker."""
    from cupy.cuda import compiler
    from cupy_backends.cuda.libs import nvrtc
    from woof import hip_backend, numerics
    target = str(arch).removeprefix('sm_')
    if compiler._use_ptx or int(target) not in nvrtc.getSupportedArchs():
        raise ValueError('HIP NVIDIA plan requires an explicit supported cubin target')
    selector = compiler._get_arch_for_options_for_nvrtc

    def explicit(selected=None):
        if str(selected) != target:
            raise ValueError('HIP NVIDIA worker target changed')
        return '-arch=sm_' + target, 'cubin'

    hip_backend.install(compiler)
    compiler._get_arch_for_options_for_nvrtc = explicit
    try:
        with numerics.preserve_final_options():
            return compile(arguments['source'], tuple(arguments['options']), target,
                           arguments['filename'], arguments.get('name_expressions'),
                           cache_in_memory=arguments.get('cache_in_memory', False),
                           method=arguments.get('method'))
    finally:
        compiler._get_arch_for_options_for_nvrtc = selector


def compile(source, options, arch, filename, name_expressions, **kwargs):
    from woof import hip_backend
    from woof.hip_source import translate
    hip = translate(source, nvidia=True)
    # CUDA-built CuPy's headers must retain their CUDA/NVRTC branch on
    # HIP's NVIDIA target. Their AMD RTC branch includes host C++ headers
    # that NVIDIA's runtime compiler intentionally does not provide.
    hip = re.sub(r'(^\s*#\s*include\s*[<"]cupy/[^>"\n]+[>"])',
        lambda m: '#undef __HIPCC_RTC__\n#undef __HIPCC__\n'+m[0]+
        '\n#define __HIPCC__ 1\n#define __HIPCC_RTC__ 1', hip, flags=re.M)
    root = Path(hip_backend.toolchain_identity()[0]).resolve().parent.parent
    opts = (*options, '-I'+str(root/'include'),
            '-D__HIP_PLATFORM_NVIDIA__=1', '-D__HIPCC__=1', '-D__HIPCC_RTC__=1')
    if arch is not None:
        arch = str(arch).removeprefix('sm_').removeprefix('compute_')
    ident = identity()
    headers = [hip_backend.compiler_dependencies(str(root/'bin/hipcc')),
               *hip_backend.header_identity(opts)]
    key = hashlib.sha256(json.dumps([hip, opts, arch, filename, name_expressions,
        ident, headers, hashlib.sha256(Path(__file__).read_bytes()).hexdigest()],
        sort_keys=True).encode()).hexdigest()
    cache = Path(os.environ.get('CUPY_CACHE_DIR', str(Path.home()/'.cache/cupy')))/'gpuwm-hiprtc-nvidia'
    cache.mkdir(parents=True, exist_ok=True)
    image, meta = cache/(key+'.image'), cache/(key+'.json')
    if image.exists() and meta.exists():
        row = json.loads(meta.read_text()); blob = image.read_bytes()
        if hashlib.sha256(blob).hexdigest() == row['image_sha256']:
            return blob, row['mapping']
    original = hip_backend._ORIGINAL_DIRECT
    if original is None:
        raise RuntimeError('HIP runtime compiler was not installed')
    blob, mapping = original(hip, opts, arch, filename, name_expressions, **kwargs)
    row = dict(compiler=ident, headers=headers, options=opts, mapping=mapping,
               source_sha256=hashlib.sha256(source.encode()).hexdigest(),
               translated_sha256=hashlib.sha256(hip.encode()).hexdigest(),
               image_sha256=hashlib.sha256(blob).hexdigest())
    with tempfile.NamedTemporaryFile(dir=cache, delete=False) as stream:
        stream.write(blob); temporary=stream.name
    os.replace(temporary, image)
    meta.write_text(json.dumps(row, indent=2))
    return blob, mapping
