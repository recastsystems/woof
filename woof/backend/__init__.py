"""Process-wide forecast execution backend, shared by both model engines.

Set WOOF_BACKEND=amd before importing forecast modules to opt into the
experimental AMD backend. CUDA is the default. Selection freezes on first use.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
import sys
import threading
from types import ModuleType


@dataclass(frozen=True)
class Backend:
    name: str
    xp: object
    is_host: bool

    def synchronize(self):
        if not self.is_host:
            self.xp.cuda.get_current_stream().synchronize()

    def compile(self, code, options=(), name_expressions=None):
        from woof.certify.kernel_manifest import record_module, source_sha256
        module = self.xp.RawModule(code=code, options=options,
                                   name_expressions=name_expressions)
        record_module("backend-" + source_sha256(code)[:16], source=code,
                      options=options, module=module)
        return module

    def to_host(self, array):
        return self.xp.asnumpy(array)


_selected = None
_lock = threading.RLock()
_factories = {}


def register(name, factory):
    """Register an additional backend before selecting it (e.g. ROCm)."""
    with _lock:
        if name in {"auto", "cuda", "cpu", "amd"} or name in _factories:
            raise ValueError(f"backend already registered: {name}")
        _factories[name] = factory


def select(name):
    name = {"amd": "hip", "rocm": "hip", "auto": "cuda"}.get(name, name)
    with _lock:
        if name not in {"cuda"} | _factories.keys():
            raise ValueError(f"unknown forecast backend: {name}")
        if _selected is not None and name not in ("auto", _selected.name):
            raise RuntimeError(f"forecast backend already fixed to {_selected.name}")
        os.environ["WOOF_BACKEND"] = name
    return name


def _cuda():
    from woof.local_gpu import no_local_gpu
    if no_local_gpu():
        raise RuntimeError("local GPU access disabled")
    cp = importlib.import_module("cupy")
    if cp.cuda.runtime.is_hip:
        raise RuntimeError("CUDA is the default; select --backend amd for experimental ROCm execution")
    return Backend("cuda", cp, False)


def get_backend():
    global _selected
    with _lock:
        if _selected is not None:
            return _selected
        name = os.environ.get("WOOF_BACKEND", "cuda").strip().lower()
        name = {"amd": "hip", "rocm": "hip", "auto": "cuda"}.get(name, name)
        if name in _factories:
            _selected = _factories[name]()
        elif name == "cuda":
            _selected = _cuda()
        else:
            raise ValueError(f"unknown forecast backend: {name}; choose cuda or amd (experimental)")
        os.environ["WOOF_BACKEND"] = _selected.name
        return _selected


def array_module():
    return get_backend().xp


def is_numpy_setup_array(value):
    """Distinguish preprocessing storage from arrays owned by a backend."""
    import numpy as np
    return isinstance(value, np.ndarray) and not getattr(value, "_gpuwm_backend_array", False)


def current_device_id(array_module=None):
    """Current execution domain, including explicitly supplied array APIs."""
    if array_module is None:
        backend = get_backend()
        custom = getattr(backend, "current_device_id", None)
        if custom is not None:
            return int(custom())
        array_module = backend.xp
    return (0 if getattr(array_module, "_gpuwm_host_backend", False)
            else int(array_module.cuda.runtime.getDevice()))


def memory_info(*, array_module=None):
    """Available and total bytes in the selected execution domain."""
    if array_module is None:
        backend = get_backend()
        custom = getattr(backend, "memory_info", None)
        if custom is not None:
            return tuple(map(int, custom()))
        array_module, host = backend.xp, backend.is_host
    else:
        host = getattr(array_module, "_gpuwm_host_backend", False)
    if not host:
        return tuple(map(int,array_module.cuda.runtime.memGetInfo()))
    from pathlib import Path
    values={line.split(':')[0]:int(line.split()[1])*1024
            for line in Path('/proc/meminfo').read_text().splitlines()
            if line.startswith(('MemAvailable:', 'MemTotal:'))}
    return values['MemAvailable'],values['MemTotal']


def compilation_target(device=None, *, xp=None):
    """CUDA compilation architecture, or None for native CPU compilation."""
    if xp is not None:
        if getattr(getattr(xp.cuda,'runtime',None),'is_hip',False):
            from woof.hip_backend import architecture
            return architecture(xp,device)
        return (xp.cuda.Device() if device is None else xp.cuda.Device(device)).compute_capability
    backend=get_backend()
    custom = getattr(backend, "compilation_target", None)
    if custom is not None:
        return custom(device)
    if backend.is_host:return None
    if backend.name == 'hip':
        from woof.hip_backend import architecture
        return architecture(backend.xp,device)
    return backend.xp.cuda.Device(device).compute_capability


def image_target(device=None, *, xp=None):
    """Native image ISA, including its gfx or sm prefix."""
    target=compilation_target(device,xp=xp)
    if target is None:return None
    return str(target) if str(target).startswith('gfx') else 'sm_'+str(target)


def active_blocks_per_multiprocessor(function, block_size, dynamic_shared=0, *, xp=None):
    """Occupancy from the selected execution adapter's loaded kernel."""
    if xp is None:
        backend = get_backend()
        custom = getattr(backend, 'active_blocks_per_multiprocessor', None)
        if custom is not None:
            return int(custom(function, block_size, dynamic_shared))
        xp = backend.xp
    if getattr(xp.cuda.runtime, 'is_hip', False):
        from woof.hip_runtime import active_blocks_per_multiprocessor as hip_occupancy
        return hip_occupancy(function, block_size, dynamic_shared)
    return int(xp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(
        function, block_size, dynamic_shared))


def consume_backend_option(argv, *, local_backend_commands=()):
    """Allow the common backend option before or after a command word."""
    tokens=list(argv);remaining=[];i=0;command=None
    while i<len(tokens):
        token=tokens[i]
        if command in local_backend_commands:
            remaining.extend(tokens[i:])
            break
        if token=='--backend':
            if i+1==len(tokens):
                raise ValueError('--backend requires a backend name')
            select(tokens[i+1]);i+=2
        elif token.startswith('--backend='):
            select(token.split('=',1)[1]);i+=1
        else:
            remaining.append(token);i+=1
            if command is None and not token.startswith('-'):
                command=token
    if "WOOF_BACKEND" in os.environ:
        select(os.environ["WOOF_BACKEND"].strip().lower())
    return remaining


class _ArrayModule(ModuleType):
    def __getattr__(self, name):
        value = getattr(array_module(), name)
        # Backend selection is immutable. Bind each array API attribute once,
        # so the CUDA hot path calls the original CuPy function directly.
        setattr(self, name, value)
        return value


xp = _ArrayModule("woof.backend.xp")

# GPU adapters share this interface and the canonical kernel sources.
from woof.hip_backend import HipBackend
register('hip', lambda: HipBackend('hip'))
register('hip-nvidia', lambda: HipBackend('hip-nvidia'))


def __getattr__(name):
    if name in ("cuda", "compiler"):
        runtime = array_module().cuda
        return runtime if name == "cuda" else runtime.compiler
    raise AttributeError(name)
