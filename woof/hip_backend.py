"""HIP adapter for the shared woof.backend contract.

The array, BLAS, FFT, RNG, stream and memory implementation is CuPy built
against ROCm. HIP-on-NVIDIA keeps CUDA CuPy arrays but compiles the same
translated HIP source with HIP_PLATFORM=nvidia; it is a translation test,
not evidence of AMD numerical identity.
"""
from __future__ import annotations
import functools
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from woof.hip_source import translate, options_for_amd, REVISION

_HEADER_SNAPSHOTS={}

def header_identity(options):
    """Include-tree content binding, once per immutable compiler session."""
    from woof.kernel_dependencies import include_tree_dependencies,include_roots
    hipcc,_=toolchain_identity()
    root=Path(hipcc).resolve().parent.parent
    # compiler_dependencies already binds every ROCm and clang header. Do not
    # run the external-header escape heuristic over that trusted compiler
    # tree: ROCm 7's miopen/config.h defines absolute executable paths, which
    # are data strings rather than includes. External CuPy/project roots still
    # receive the full conservative include-tree check.
    implicit=[root/'include',*compiler_include_directories(root)]
    external=[p for p in include_roots(options)
              if not any(p.resolve().is_relative_to(base.resolve()) for base in implicit)]
    if not external:return []
    # Importing the kernel package writes interpreter-specific bytecode beside
    # its headers. These generated files are never C/C++ input; retain all
    # source/header files while making prebuilt bindings survive installation.
    return include_tree_dependencies(tuple('-I'+str(p) for p in external),
                                     _HEADER_SNAPSHOTS,
                                     exclude_python_bytecode=True)


def compiler_include_directories(root):
    """ROCm 6 and 7 compiler-resource layouts, including device bitcode."""
    return [p for llvm in (root/'llvm',root/'lib/llvm')
            for version in sorted((llvm/'lib/clang').glob('*'))
            for p in (version/'include',version/'lib/amdgcn/bitcode') if p.is_dir()]


def requested():
    name = os.environ.get('WOOF_BACKEND', 'cuda').strip().lower()
    return {'amd': 'hip', 'rocm': 'hip', 'auto': 'cuda'}.get(name, name)


def active():
    return requested() in ('hip', 'hip-nvidia')


def architecture(cp=None, device=None):
    if cp is None:
        import cupy as cp
    if device is None: device=cp.cuda.runtime.getDevice()
    if cp.cuda.runtime.is_hip:
        props = cp.cuda.runtime.getDeviceProperties(device)
        value = props['gcnArchName']
        detected = value.decode() if isinstance(value, bytes) else str(value)
        target = os.environ.get('WOOF_HIP_ARCH', detected)
        # A generic gfx942 code object accepts the device's reported feature
        # suffixes (for example :sramecc+:xnack-). Keep feature-specific JIT
        # by default, but let a declared, matching generic kit use its pack.
        if target.split(':', 1)[0] != detected.split(':', 1)[0]:
            raise ValueError(f'HIP target {target} does not match device {detected}')
        return target
    return 'sm_' + str(cp.cuda.Device(device).compute_capability)


def runtime_limit(limit, value=None, *, runtime=None):
    """Real HIP limits: CuPy 14's HIP deviceSetLimit shim always errors."""
    if runtime is None:
        import cupy as cp
        runtime=cp.cuda.runtime
    if not getattr(runtime,'is_hip',False):
        return runtime.deviceGetLimit(limit) if value is None else runtime.deviceSetLimit(limit,value)
    import ctypes
    root=Path(os.environ.get('ROCM_PATH',os.environ.get('ROCM_HOME','/opt/rocm')))
    library=ctypes.CDLL(str(root/'lib/libamdhip64.so'))
    if value is None:
        result=ctypes.c_size_t()
        call=library.hipDeviceGetLimit
        call.argtypes=[ctypes.POINTER(ctypes.c_size_t),ctypes.c_int]
        status=call(ctypes.byref(result),int(limit))
    else:
        call=library.hipDeviceSetLimit
        call.argtypes=[ctypes.c_int,ctypes.c_size_t]
        status=call(int(limit),int(value))
    if status: raise runtime.CUDARuntimeError(status)
    return result.value if value is None else None


@functools.lru_cache(maxsize=8)
def toolchain_identity(hipcc=None):
    hipcc = hipcc or os.environ.get('HIPCC') or shutil.which('hipcc')
    if not hipcc:
        raise RuntimeError('HIP compilation needs hipcc on PATH or HIPCC pointing to the ROCm compiler')
    env = dict(os.environ, HIP_PLATFORM='amd', HCC_AMDGPU_TARGET='gfx942')
    result = subprocess.run([hipcc, '--version'], env=env, capture_output=True, text=True, check=True)
    return hipcc, result.stdout


@functools.lru_cache(maxsize=8)
def compiler_dependencies(hipcc):
    root=Path(hipcc).resolve().parent.parent
    digest=hashlib.sha256()
    paths=[Path(hipcc).resolve(),root/'llvm/bin/clang++',root/'llvm/bin/clang-offload-bundler']
    for directory in (root/'include',*compiler_include_directories(root),root/'amdgcn/bitcode'):
        if directory.is_dir(): paths.extend(sorted(p for p in directory.rglob('*') if p.is_file()))
    for path in paths:
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return dict(compiler_and_implicit_inputs_sha256=digest.hexdigest())

@functools.lru_cache(maxsize=8)
def nvidia_banner(compiler):
    return subprocess.run([compiler,'--version'],env=dict(os.environ,HIP_PLATFORM='nvidia'),
        capture_output=True,text=True,check=True).stdout


@functools.lru_cache(maxsize=8)
def nvidia_compiler_identity(compiler):
    """Bind the NVIDIA compiler actually used by the HIP translation test."""
    executable = Path(shutil.which(compiler) or compiler).resolve()
    root = executable.parent.parent
    paths = [executable, root/'bin/ptxas', root/'nvvm/bin/cicc']
    for directory in (root/'include', root/'nvvm/libdevice'):
        if directory.is_dir():
            paths.extend(sorted(p for p in directory.rglob('*') if p.is_file()))
    digest = hashlib.sha256()
    for path in paths:
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return dict(version=nvidia_banner(compiler), library_sha256=digest.hexdigest(),
                compiler=str(executable))


def runtime_compiler_identity():
    """Compatibility name for the selected HIP translation compiler."""
    return compiler_runtime_identity()


def code_object_resources(image, hipcc=None):
    """Actual AMD code-object metadata; no CUDA frame estimates or GPU needed."""
    import re
    hipcc,_=toolchain_identity(hipcc)
    readelf=Path(hipcc).resolve().parent.parent/'llvm/bin/llvm-readelf'
    notes=subprocess.run([str(readelf),'--notes',str(image)],check=True,
                         capture_output=True,text=True).stdout
    def values(field):return [int(x) for x in re.findall(r'\.'+field+r':\s*(\d+)',notes)]
    private=values('private_segment_fixed_size')
    return dict(max_private_segment_bytes=max(private,default=0),
                max_group_segment_bytes=max(values('group_segment_fixed_size'),default=0),
                wavefront_sizes=sorted(set(values('wavefront_size'))),kernels=len(private))


def cached_private_frame_bound():
    """Conservative maximum across compiled AMD images in this process cache.

    This is compiler scratch metadata, not a measured device-memory peak.
    An absent entry cannot establish that an uncompiled kernel uses no scratch.
    """
    root=Path(os.environ.get('CUPY_CACHE_DIR',str(Path.home()/'.cache/cupy')))/'gpuwm-hip'
    maximum=0
    for path in root.glob('*.hsaco.json'):
        try:
            record=json.loads(path.read_text())
            maximum=max(maximum,int(record.get('resources',{}).get('max_private_segment_bytes',0)))
        except (OSError,ValueError,TypeError):continue
    return maximum

def compile_aot(source, options, target, *, output=None, hipcc=None, cache_only=False):
    """Build a source-bound ELF for gfx* or sm_* without opening a GPU."""
    hipcc, version = toolchain_identity(hipcc)
    amd = target.startswith('gfx')
    hip = translate(source, nvidia=not amd)
    if not amd and not target.startswith('sm_'):
        raise ValueError('HIP target must be a gfx architecture or sm_ architecture')
    opts = options_for_amd(options) if amd else tuple(o for o in options
        if not (o.startswith('-I') and ('nvidia/cu13/' in o or '/cuda13/' in o)))
    compiler = hipcc if amd else os.environ.get('HIPCC_NVIDIA', hipcc)
    if not amd:
        version += nvidia_banner(compiler)
    headers=[compiler_dependencies(hipcc),*header_identity(opts)]
    key = hashlib.sha256(json.dumps([hip, opts, target, version, headers, REVISION]).encode()).hexdigest()
    directory = Path(os.environ.get('CUPY_CACHE_DIR', str(Path.home()/'.cache'/'cupy')))/'gpuwm-hip'
    directory.mkdir(parents=True, exist_ok=True)
    image = Path(output) if output else directory/(key+('.hsaco' if amd else '.cubin'))
    meta = image.with_suffix(image.suffix+'.json')
    if image.exists() and meta.exists():
        record = json.loads(meta.read_text())
        if record.get('key') == key and hashlib.sha256(image.read_bytes()).hexdigest() == record.get('image_sha256'):
            return image.read_bytes(), record
    if amd and os.environ.get('WOOF_HIP_PREBUILT'):
        from woof.hip_prebuilt import binding, lookup
        value = binding(hashlib.sha256(source.encode()).hexdigest(),
            hashlib.sha256(hip.encode()).hexdigest(), target, opts, headers, REVISION)
        packed = lookup(value)
        if packed is not None:
            blob, record = packed
            record.update(key=key, compiler=version, headers=headers, options=list(opts))
            image.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=image.parent, delete=False) as handle:
                handle.write(blob); temporary=handle.name
            os.replace(temporary, image)
            meta.write_text(json.dumps(record, indent=2))
            return blob, record
    canonical=directory/(key+('.hsaco' if amd else '.cubin'))
    canonical_meta=canonical.with_suffix(canonical.suffix+'.json')
    if output and canonical.exists() and canonical_meta.exists():
        record=json.loads(canonical_meta.read_text());blob=canonical.read_bytes()
        if record.get('key')==key and hashlib.sha256(blob).hexdigest()==record.get('image_sha256'):
            image.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=image.parent,delete=False) as handle:
                handle.write(blob);temporary=handle.name
            os.replace(temporary,image);meta.write_text(json.dumps(record,indent=2))
            return blob,record
    if cache_only:return None,None
    image.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='gpuwm-hip-') as tmp:
        src = Path(tmp)/'kernel.hip.cpp'; src.write_text(hip)
        built = Path(tmp)/'kernel.image'
        env = dict(os.environ, HIP_PLATFORM='amd' if amd else 'nvidia')
        flags = ['--genco','--offload-arch='+target] if amd else ['--cubin','-arch='+target, '-x', 'cu']
        include = str(Path(hipcc).resolve().parent.parent/'include')
        if not amd:
            flags += ['-I'+include, '-D__HIP_PLATFORM_NVIDIA__=1', '-D__HIPCC__=1', '-allow-unsupported-compiler']
        if not amd: flags.remove('-I'+include)
        command = [compiler, *flags, '-O2', *opts, *(['-I'+include] if not amd else []), str(src), '-o', str(built)]
        proc = subprocess.run(command, env=env, capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError('HIP compile failed: '+ ' '.join(command)+'\n'+proc.stdout+proc.stderr)
        if amd:
            bundler = str(Path(hipcc).resolve().parent.parent/'llvm/bin/clang-offload-bundler')
            if built.read_bytes().startswith(b'__CLANG_OFFLOAD_BUNDLE__'):
                extracted = Path(tmp)/'device.hsaco'
                subprocess.run([bundler, '--unbundle', '--type=o',
                                '--targets=hipv4-amdgcn-amd-amdhsa--'+target,
                                '--input='+str(built), '--output='+str(extracted)],check=True,capture_output=True)
                built = extracted
        blob = built.read_bytes()
        if blob[:4] != b'\x7fELF': raise RuntimeError('HIP compiler did not produce an ELF device image')
        record = dict(key=key, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                      translated_sha256=hashlib.sha256(hip.encode()).hexdigest(),
                      image_sha256=hashlib.sha256(blob).hexdigest(), arch=target,
                      compiler=version, headers=headers, options=list(opts), command=command, translator=REVISION)
        if amd:record['resources']=code_object_resources(built,hipcc)
        # Atomic image publication; a partial/missing manifest is a cache miss.
        with tempfile.NamedTemporaryFile(dir=image.parent, delete=False) as f:
            f.write(blob); temporary=Path(f.name)
        os.replace(temporary,image)
        meta.write_text(json.dumps(record,indent=2))
        if output:
            canonical=directory/(key+('.hsaco' if amd else '.cubin'))
            shutil.copyfile(image,canonical)
            canonical.with_suffix(canonical.suffix+'.json').write_text(json.dumps(record,indent=2))
        return blob, record


_ORIGINAL_DIRECT = None


@functools.lru_cache(maxsize=32)
def wall_clock_rate_khz(device):
    import ctypes
    # Public hipDeviceAttribute_t AMD extension, stable in the pinned 6.4 API.
    root=Path(os.environ.get('ROCM_PATH',os.environ.get('ROCM_HOME','/opt/rocm')))
    library=ctypes.CDLL(str(root/'lib/libamdhip64.so'))
    function=library.hipDeviceGetAttribute
    function.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.c_int]
    function.restype=ctypes.c_int
    rate=ctypes.c_int()
    error=function(ctypes.byref(rate),10017,int(device))
    if error or rate.value<=0:raise RuntimeError('HIP wall clock rate query failed: '+str(error))
    return rate.value


def initialize_module(module):
    import ctypes,cupy as cp
    if cp.cuda.runtime.is_hip:
        pointer=module.get_global_var('gpuwm_wall_clock_khz')
        rate=ctypes.c_uint64(wall_clock_rate_khz(cp.cuda.runtime.getDevice()))
        cp.cuda.runtime.memcpy(int(pointer),ctypes.addressof(rate),ctypes.sizeof(rate),
                               cp.cuda.runtime.memcpyHostToDevice)
    return module

def compile_source(source, options=(), arch=None, filename='kern.cu', name_expressions=None,
                   log_stream=None, cache_in_memory=False, jitify=None, method=None):
    import cupy as cp
    from woof.backend import get_backend
    get_backend() # Freeze the shared selection before the first compilation.
    from woof.numerics import effective_options
    opts = effective_options(options)
    if requested() == 'hip-nvidia':
        policy = os.environ.get('WOOF_HIP_NVIDIA_COMPILER', 'aot')
        if policy == 'nvrtc':
            from woof.hip_nvidia_rtc import compile
            return compile(source, opts, arch, filename, name_expressions,
                           log_stream=log_stream, cache_in_memory=cache_in_memory,
                           jitify=jitify, method=method)
        if policy != 'aot':
            raise ValueError('WOOF_HIP_NVIDIA_COMPILER must be aot or nvrtc')
        if name_expressions:
            raise ValueError('HIP-on-NVIDIA AOT cannot resolve template name expressions; use extern C kernel names')
        target=arch or architecture(cp)
        if str(target).isdigit(): target='sm_'+str(target)
        blob, _ = compile_aot(source, opts, target)
        return blob, None
    if not cp.cuda.runtime.is_hip:
        raise RuntimeError('WOOF_BACKEND=hip needs a ROCm-built CuPy; this CuPy is compiled for CUDA')
    if jitify or method not in (None,'ptx'):
        raise ValueError('HIPRTC does not support this Jitify/LTO request')
    # Legacy CUDA callers may still pass CuPy's numeric compute capability.
    # HIPRTC/HIPCC must always receive the AMD ISA, never sm_94 or compute_94.
    target=architecture(cp) if not arch or not str(arch).startswith('gfx') else arch
    policy=os.environ.get('WOOF_HIP_COMPILER','auto')
    if policy not in ('auto','hipcc','hiprtc'):raise ValueError('WOOF_HIP_COMPILER must be auto, hipcc or hiprtc')
    if not name_expressions and policy!='hiprtc':
        blob,_=compile_aot(source,opts,target,cache_only=policy=='auto')
        if blob is not None:return blob,None
    elif policy=='hipcc':
        raise ValueError('HIPCC AOT templates require lowered names; use the HIPRTC route')
    from woof.hiprtc import compile
    return compile(source,opts,target,filename,name_expressions)


def install(compiler=None):
    """Bridge both legacy CuPy routes before their cache keys are formed."""
    global _ORIGINAL_DIRECT
    if not active(): return
    import sys
    os.environ.setdefault('WOOF_PYTHON',sys.executable)
    os.environ.setdefault('WOOF_RUST_BACKEND',requested())
    if compiler is None:
        from cupy.cuda import compiler
    if getattr(compiler, '_gpuwm_hip_backend', False): return
    _ORIGINAL_DIRECT = compiler._compile_using_nvrtc_no_warning
    old_cache = compiler._compile_module_with_cache

    @functools.wraps(old_cache)
    def cached(source, options=(), arch=None, cache_dir=None, extra_source=None,
               backend='nvrtc', enable_cooperative_groups=False, name_expressions=None,
               log_stream=None, cache_in_memory=False, jitify=False, **kwargs):
        if backend != 'nvrtc' or enable_cooperative_groups or kwargs:
            raise ValueError('HIP does not support this linked/cooperative compiler request')
        from cupy.cuda import function
        # CuPy's CUDA _compile_with_cache_cuda appends this option after
        # the caller's tuple. This bridge bypasses that function; reproduce
        # its arithmetic here, while direct NVRTC callers keep their tuple.
        options = tuple(options) + ('-ftz=true',)
        image, mapping = compiler._compile_using_nvrtc_no_warning(
            source, options, arch, 'raw.hip.cpp', name_expressions,
            log_stream, cache_in_memory, jitify)
        module = function.Module(); module.load(image);initialize_module(module)
        if mapping: module._set_mapping(mapping)
        return module
    compiler._compile_module_with_cache = cached
    compiler._compile_using_nvrtc_no_warning = compile_source
    compiler._gpuwm_hip_backend = True


def arm():
    # The existing numerics lazy compiler hook calls install after CuPy loads.
    import sys
    if 'cupy.cuda.compiler' in sys.modules: install(sys.modules['cupy.cuda.compiler'])


class HipBackend:
    is_host = False
    def __init__(self, name='hip'):
        if name not in ('hip','hip-nvidia'): raise ValueError(name)
        self.name=name
        import cupy as cp
        if bool(cp.cuda.runtime.is_hip) != (name=='hip'):
            raise RuntimeError(name+' requires '+('ROCm' if name=='hip' else 'CUDA')+' CuPy')
        self.xp=cp
        if cp.cuda.runtime.getDeviceCount()<1:raise RuntimeError(name+' reports no GPUs')
        from woof.hip_runtime import initialize_device_attributes
        self.device_attribute_corrections = initialize_device_attributes(cp)
        install()
    def synchronize(self): self.xp.cuda.get_current_stream().synchronize()
    def to_host(self, array): return self.xp.asnumpy(array)
    def memory_info(self): return self.xp.cuda.runtime.memGetInfo()
    def current_device_id(self): return self.xp.cuda.runtime.getDevice()
    def compilation_target(self, device=None):
        target = architecture(self.xp, device)
        return target if self.name == 'hip' else target.removeprefix('sm_')
    def compile(self, code, options=(), name_expressions=None):
        from cupy.cuda import function
        image,mapping=compile_source(code,options,name_expressions=name_expressions)
        module=function.Module(); module.load(image);initialize_module(module)
        if mapping: module._set_mapping(mapping)
        return module


def device_info():
    import cupy as cp
    p=cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    name=p['name']; name=name.decode() if isinstance(name,bytes) else str(name)
    free,total=cp.cuda.runtime.memGetInfo()
    rocm_version = None
    if cp.cuda.runtime.is_hip:
        root = Path(os.environ.get('ROCM_PATH', os.environ.get('ROCM_HOME', '/opt/rocm')))
        for version_file in (root / '.info/version', root / '.info/version-dev'):
            if version_file.is_file():
                rocm_version = version_file.read_text().strip()
                break
    from woof.hip_runtime import device_attribute
    wave_size = device_attribute(cp, 'WarpSize')
    return dict(backend='amd' if cp.cuda.runtime.is_hip else requested(),name=name,
                rocm_version=rocm_version,gfx_target=architecture(cp),wave_size=wave_size,
                arch=architecture(cp),warp_size=wave_size,free_bytes=int(free),
                total_bytes=int(total),runtime=int(cp.cuda.runtime.runtimeGetVersion()),
                driver=int(cp.cuda.runtime.driverGetVersion()),cupy=cp.__version__)


def supports_graphs():
    import cupy as cp
    # CuPy currently documents stream capture as unsupported on HIP. Existing
    # callers already have a normal stream fallback; use that on ROCm.
    return not cp.cuda.runtime.is_hip


def compile_fingerprint():
    """A distinct, truthful HIP certificate schema (never NVRTC placeholders)."""
    import cupy as cp
    import numpy as np
    info=device_info()
    compiler=os.environ.get('HIPCC_NVIDIA') if requested()=='hip-nvidia' else None
    _,banner=toolchain_identity()
    if compiler:
        banner += subprocess.run([compiler,'--version'],capture_output=True,text=True,check=True).stdout
    rtc=compiler_runtime_identity()
    return dict(backend=requested(),compiler_build=banner.strip(),
        compiler_library_sha256=rtc['library_sha256'],
        portable_header_sha256=hashlib.sha256(Path(__file__).with_name('hip_portable.h').read_bytes()).hexdigest(),
        translator_sha256=hashlib.sha256(Path(__file__).with_name('hip_source.py').read_bytes()).hexdigest(),
        device_architecture=info['arch'],warp_size=str(info['warp_size']),
        gpu_runtime_version=str(info['runtime']),gpu_driver_version=str(info['driver']),
        cupy_version=cp.__version__,numpy_version=np.__version__)


def compiler_binding():
    """Off-card identity for the existing parallel compilation plan."""
    import cupy
    hipcc,banner=toolchain_identity()
    result = dict(backend=requested(),cupy=cupy.__version__,hipcc=banner,
        inputs=compiler_dependencies(hipcc),hiprtc=compiler_runtime_identity(),
        translator=hashlib.sha256(Path(__file__).with_name('hip_source.py').read_bytes()).hexdigest(),
        portable_header=hashlib.sha256(Path(__file__).with_name('hip_portable.h').read_bytes()).hexdigest())
    if requested() == 'hip-nvidia':
        # Earlier parallel workers ignored the selected NVIDIA compiler.
        # Never accept their images under the corrected compiler contract.
        result['worker_contract_sha256'] = hashlib.sha256(b''.join(
            Path(__file__).with_name(name).read_bytes()
            for name in ('kernel_compile_worker.py', 'hip_nvidia_rtc.py'))).hexdigest()
    return result


def compiler_runtime_identity():
    """Bind the compiler used by this mode; HIP-on-NVIDIA never uses HIPRTC."""
    if requested() == 'hip-nvidia':
        if os.environ.get('WOOF_HIP_NVIDIA_COMPILER', 'aot') == 'nvrtc':
            from woof.hip_nvidia_rtc import identity
            return identity()
        hipcc, _ = toolchain_identity()
        compiler = os.environ.get('HIPCC_NVIDIA', hipcc)
        nvcc = Path(os.environ.get('CUDA_PATH', os.environ.get('CUDA_HOME', '/usr/local/cuda'))) / 'bin/nvcc'
        return dict(nvidia_compiler_identity(str(nvcc)),
                    mode='hipcc-nvidia-aot', launcher=compiler)
    from woof.hiprtc import identity
    return identity()


def compile_shared(source, target=None):
    """Compile canonical native host launchers against the selected HIP runtime."""
    target=target or architecture()
    hipcc,banner=toolchain_identity();amd=target.startswith('gfx')
    compiler=hipcc if amd else os.environ.get('HIPCC_NVIDIA',hipcc)
    hip=translate(source)
    identity=[hip,target,banner,compiler_dependencies(hipcc)]
    if not amd:identity.append(nvidia_banner(compiler))
    key=hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    cache=Path(os.environ.get('CUPY_CACHE_DIR',str(Path.home()/'.cache/cupy')))/'gpuwm-hip'
    cache.mkdir(parents=True,exist_ok=True);image=cache/(key+'.so');meta=image.with_suffix('.so.json')
    import fcntl
    with (cache/(key+'.lock')).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if image.exists() and meta.exists() and hashlib.sha256(image.read_bytes()).hexdigest()==json.loads(meta.read_text()).get('image_sha256'):
            return image
        with tempfile.TemporaryDirectory(prefix='gpuwm-hip-host-') as tmp:
            path=Path(tmp)/'host.hip.cpp';path.write_text(hip);output=Path(tmp)/'host.so'
            flags=['--offload-arch='+target,'-fPIC','-fhip-fp32-correctly-rounded-divide-sqrt'] if amd else ['-arch='+target,'-x','cu','-Xcompiler=-fPIC','-allow-unsupported-compiler','-I'+str(Path(hipcc).parent.parent/'include'),'-D__HIP_PLATFORM_NVIDIA__=1','-D__HIPCC__=1']
            command=[compiler,'--shared','-O3','-std=c++17',*flags,str(path),'-o',str(output)]
            result=subprocess.run(command,env=dict(os.environ,HIP_PLATFORM='amd' if amd else 'nvidia'),capture_output=True,text=True)
            if result.returncode:raise RuntimeError('HIP shared compile failed: '+' '.join(command)+'\n'+result.stdout+result.stderr)
            blob=output.read_bytes()
            with tempfile.NamedTemporaryFile(dir=cache,delete=False) as handle:handle.write(blob);temporary=handle.name
            os.replace(temporary,image)
            meta.write_text(json.dumps(dict(key=key,arch=target,command=command,image_sha256=hashlib.sha256(blob).hexdigest()),indent=2))
    return image
