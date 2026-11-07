"""Scope existing physics GPU caches to one resident rank and CUDA device.

Original function code, compiler options, tables and field arithmetic stay
unchanged. Private cache namespaces prevent simultaneous rank streams from
sharing the legacy shortwave workspace or another device's CUDA handles.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import functools
import importlib
import sys
import threading
from types import FunctionType, ModuleType


_GROUPS = {
    "legacy": ("woof.core.rrtmg_legacy", ("_cuda_sw",), {"_CUDA_SW_CACHE": lambda: None}),
    "longwave": ("woof.core.rrtmg_lw", ("_gpu_module", "_gpu_kernel", "gpu_preflight"),
                 {"_GPU_MODULE": lambda: None, "_GPU_KERNELS": dict, "_GPU_PREFLIGHTED": lambda: False}),
    "mcica": ("woof.core.rrtmg_mcica", ("_mcica_gpu_module", "_mcica_gpu_kernel", "mcica_gpu_preflight"),
              {"_MCICA_GPU_MODULE": lambda: None, "_MCICA_GPU_KERNELS": dict,
               "_MCICA_GPU_PREFLIGHTED": lambda: False}),
    "prep": ("woof.core.rrtmg_legacy_prep", ("_tables",), {"_XP_TABLES": dict}),
    "kernels": ("woof.core.kernels", ("load_module", "load_module_int_defines", "get_kernel", "get_kernel_int_defines"), {}),
    # These original factories retain arrays or per-device initialization
    # with keys that omit a card, or mutable scratch that omits a rank.
    "reflectivity": ("woof.core.refl", ("_device_tables",), {}),
    "myjsfc": ("woof.core.myjsfc", ("_device_tables",), {}),
    "noah": ("woof.core.noah", ("_device_tables",), {"_DEVICE_TABLES": dict}),
    "milbrandt": ("woof.core.milbrandt2_constants", ("ck_vector_device",), {"_CK_DEVICE": lambda: None}),
    "rrtm_taumol": ("woof.core.rrtm_taumol", ("_device_tables",), {"_DEVICE_CACHE": dict}),
    "rrtm_lw": ("woof.core.rrtm_lw", ("_device_constant",), {"_DEVICE_CONSTANT_CACHE": dict}),
    "rrtmgp": ("woof.core.rrtmgp", ("_mcica_jump_tables", "_chunk_scratch", "_rte_sm_count"),
               {"_CHUNK_SCRATCH": dict}),
    "jacobi": ("woof.core.jacobi_eigh", ("_kernel",), {}),
    "mynn_flags": ("woof.core.mynn_pbl_runtime", ("_validity_flags",), {"_VALIDITY_FLAGS": dict}),
    # The engine publishes _ADVANCE before _CLEAR. Concurrent first calls
    # can observe a half-built pair, including after checkpoint recovery.
    "cumulus_clock": ("woof.core.cumulus_clock", ("_kernels",),
                      {"_ADVANCE": lambda: None, "_CLEAR": lambda: None}),
}


#: Entrypoints wrapped by cuda_cache(maxsize=None) with neither stream
#: nor ready semantics: module and function handles keyed by device only.
_DEVICE_KEYED_FACTORIES = frozenset({
    ("kernels", "load_module"), ("kernels", "load_module_int_defines"),
    ("kernels", "get_kernel"), ("kernels", "get_kernel_int_defines"),
})


class PhysicsResourceScopeError(RuntimeError):
    """Resource scopes must outlive every rank operation that uses them."""


def _fresh_cache(name, original, initialize):
    # Device-keyed engine caches retain the same factory code but replace a
    # singleton module and boolean preflight flag with dict/set containers.
    # Resetting those to None/False makes every resident rank fail preflight.
    if name in ("_GPU_MODULE", "_MCICA_GPU_MODULE") and isinstance(original, dict):
        return {}
    if name in ("_GPU_PREFLIGHTED", "_MCICA_GPU_PREFLIGHTED"):
        if isinstance(original, set):
            return set()
        if isinstance(original, bool):
            return False
        raise PhysicsResourceScopeError(
            f"{name} has an unrecognized preflight cache type; prevents replacing a device census with an invalid singleton")
    return initialize()


class _AttributeProxy:
    """Forward attributes without modifying a shared module or class."""

    def __init__(self, target, **overrides):
        self._target = target
        self.__dict__.update(overrides)

    def __getattr__(self, name):
        # Bind the forwarded attribute on first use: the targets are modules
        # (cupy, cupy.cuda, cupy.cuda.runtime, a physics module) whose
        # attributes do not change, and the shortwave engine resolves cp.*
        # thousands of times per radiation call through this proxy.
        value = getattr(self._target, name)
        if callable(value) or isinstance(value, ModuleType):
            self.__dict__[name] = value
        return value


class _ShortwaveRuntimeProxy(_AttributeProxy):
    """Narrow the audited SW end fence to its owning rank's current stream."""

    def __init__(self, cp, scope, key):
        super().__init__(cp.cuda.runtime)
        self._cp, self._scope, self._key = cp, scope, key

    def deviceSynchronize(self):
        # Eager uploads finish on the coordinator before workers use the
        # engine; lazy uploads precede kernels on the same current stream.
        # The rank context selects one compute stream, and halo
        # receive events join copy-stream work onto it before consumption.
        # SW kernels and CuPy copies retain their original stream ordering.
        if self._scope.restored or self._scope.context.get() != self._key:
            raise PhysicsResourceScopeError("shortwave completion requires its owning active rank scope")
        if int(self._cp.cuda.runtime.getDevice()) != self._key[1]:
            raise PhysicsResourceScopeError("shortwave completion CUDA device differs from its rank scope")
        self._cp.cuda.get_current_stream().synchronize()


def _shortwave_module_proxy(module, scope, key):
    original = module.CudaSW

    def construct(*args, **kwargs):
        # The original constructor and every inherited method execute their
        # original code. Only this privately cached instance sees the proxy.
        engine = original(*args, **kwargs)
        cp = engine.cp
        runtime = _ShortwaveRuntimeProxy(cp, scope, key)
        engine.cp = _AttributeProxy(cp, cuda=_AttributeProxy(cp.cuda, runtime=runtime))
        return engine

    return _AttributeProxy(module, CudaSW=construct)


class _PhysicsResourceScope:
    def __init__(self, modules):
        self.context = ContextVar("hex_samehost_physics_rank_device", default=None)
        self.lock = threading.RLock()
        self.groups = {}
        self.states = {}
        self.active_contexts = 0
        self.restored = False
        self.aliases_patched = []
        self.aliases_restored = []
        self.dispatch_aliases = {}
        for name, (_module_name, functions, initializers) in _GROUPS.items():
            module = modules[name]
            originals = {function: getattr(module, function) for function in functions}
            for function, original in originals.items():
                base = getattr(original, "__wrapped__", original)
                if not isinstance(base, FunctionType):
                    raise PhysicsResourceScopeError(f"{module.__name__}.{function} is not the expected Python cache factory")
                if getattr(original, "_samehost_physics_scope", False):
                    raise PhysicsResourceScopeError("physics resource scope is already installed")
            for slot in initializers:
                if not hasattr(module, slot):
                    raise PhysicsResourceScopeError(f"{module.__name__} lacks the audited GPU cache {slot}")
            self.groups[name] = {"module": module, "originals": originals,
                                 "globals": dict(vars(module)), "initializers": initializers}

    @contextmanager
    def rank(self, rank, device):
        if self.restored:
            raise PhysicsResourceScopeError("a restored physics resource scope cannot be reused")
        key = (int(rank), int(device))
        if min(key) < 0:
            raise PhysicsResourceScopeError("rank and device ordinals must be nonnegative")
        token = self.context.set(key)
        with self.lock:
            self.active_contexts += 1
        try:
            yield
        finally:
            self.context.reset(token)
            with self.lock:
                self.active_contexts -= 1

    def _state(self, group, key):
        identity = (group, *key)
        # Entries are published once and never removed while workers run.
        # Existing factories need no process-wide lock on every launch.
        state = self.states.get(identity)
        if state is not None:
            return state
        with self.lock:
            state = self.states.get(identity)
            if state is not None:
                return state
            definition = self.groups[group]
            # Snapshots precede entrypoint installation. Imported aliases
            # inside them must use dispatch too, including calls into a
            # different cache group. Same-group calls are replaced below.
            namespace = {name: self.dispatch_aliases.get(id(value), value)
                         for name, value in definition["globals"].items()}
            for name, initialize in definition["initializers"].items():
                namespace[name] = _fresh_cache(name, definition["globals"][name], initialize)
            shortwave_stream_fence = group == "legacy" and "_sw" in namespace
            if shortwave_stream_fence:
                # gpuwm 2.8 _cuda_sw resolves this global module attribute.
                # A global CuPy shim would also change LW/McICA and other
                # ranks; the private namespace substitutes only construction.
                namespace["_sw"] = _shortwave_module_proxy(namespace["_sw"], self, key)
            functions = {}
            for name, original in definition["originals"].items():
                base = getattr(original, "__wrapped__", original)
                clone = FunctionType(base.__code__, namespace, base.__name__, base.__defaults__, base.__closure__)
                clone.__kwdefaults__ = base.__kwdefaults__
                functools.update_wrapper(clone, base, updated=())
                if hasattr(original, "cache_parameters"):
                    clone = functools.lru_cache(**original.cache_parameters())(clone)
                elif (group, name) in _DEVICE_KEYED_FACTORIES:
                    # woof.core.device_cache.cuda_cache(maxsize=None) keys its
                    # lru on the current device; it carries cache_info but no
                    # cache_parameters, so the clone above was uncached.  The
                    # breakage this prevents (nsys, RTX PRO 6000 host, 30 km on 2 cards,
                    # 2026-10-11): every rank recompiled-from-cache, loaded and
                    # then unloaded the sfclay module on every step
                    # (cuModuleLoadData 1.3 ms + a blocking cuModuleUnload
                    # 9-10 ms + 5 ms of source lookup per rank per step).  A
                    # rank scope has one device, so an lru on the arguments
                    # is the same key space.
                    clone = functools.lru_cache(maxsize=None)(clone)
                functions[name] = clone
            # Calls between cache factories use the same private namespace.
            namespace.update(functions)
            state = {"globals": namespace, "functions": functions,
                     "shortwave_stream_fence": shortwave_stream_fence}
            self.states[identity] = state
            return state

    def dispatch(self, group, function, args, kwargs):
        key = self.context.get()
        original = self.groups[group]["originals"][function]
        selected = original if key is None else self._state(group, key)["functions"][function]
        return selected(*args, **kwargs)

    def receipt(self):
        with self.lock:
            rows = []
            for (group, rank, device), state in sorted(self.states.items()):
                cache_counts = {}
                for name in self.groups[group]["initializers"]:
                    value = state["globals"][name]
                    cache_counts[name] = (len(value) if isinstance(value, (dict, tuple, set))
                                          else int(value is not None and value is not False))
                lru = {}
                for name, function in state["functions"].items():
                    if hasattr(function, "cache_info"):
                        info = function.cache_info()
                        lru[name] = {"hits": info.hits, "misses": info.misses, "entries": info.currsize}
                rows.append({"group": group, "rank": rank, "device": device,
                             "cache_counts": cache_counts, "lru": lru,
                             "shortwave_completion_on_current_stream": state["shortwave_stream_fence"]})
            return {"schema": "mpas-port.samehost-physics-resource-scope/v1",
                    "key": "resident rank and CUDA device",
                    "groups": {name: {"module": definition["module"].__name__,
                                      "entrypoints": list(definition["originals"]),
                                      "private_gpu_cache_slots": list(definition["initializers"])}
                               for name, definition in self.groups.items()},
                    "scopes": rows, "active_rank_contexts": self.active_contexts,
                    "module_entrypoints_restored": self.restored,
                    "existing_aliases_patched": list(self.aliases_patched),
                    "reader_aliases_restored": list(self.aliases_restored),
                    "numerical_function_code_unchanged": True,
                    "cpu_coefficient_tables_shared": True}


@contextmanager
def scoped_physics_resources(*, modules=None):
    """Install on the coordinator; restore after every rank stream is drained.

    ``rank(rank, device)`` is required around construction and every worker
    call. Unscoped calls retain the original process cache behavior.
    """
    if modules is None:
        modules = {name: importlib.import_module(module_name)
                   for name, (module_name, _functions, _initializers) in _GROUPS.items()}
    scope = _PhysicsResourceScope(modules)
    changes = []
    reverse = {}
    replacements = {}
    try:
        for group, definition in scope.groups.items():
            module = definition["module"]
            for name, original in definition["originals"].items():
                def make_dispatch(group, name, original):
                    @functools.wraps(original, updated=())
                    def invoke(*args, **kwargs):
                        return scope.dispatch(group, name, args, kwargs)
                    invoke._samehost_physics_scope = True
                    return invoke
                invoke = make_dispatch(group, name, original)
                changes.append((module, name, original))
                setattr(module, name, invoke)
                replacements[id(original)] = invoke
                reverse[id(invoke)] = original
        scope.dispatch_aliases.update(replacements)
        for module_name, module in tuple(sys.modules.items()):
            if module is None or not module_name.startswith("woof."):
                continue
            for name, value in tuple(vars(module).items()):
                replacement = replacements.get(id(value))
                if replacement is not None:
                    changes.append((module, name, value))
                    setattr(module, name, replacement)
                    scope.aliases_patched.append(f"{module_name}.{name}")
        yield scope
    finally:
        if scope.active_contexts:
            raise PhysicsResourceScopeError("physics workers must drain before cache entrypoints are restored")
        for module_name, module in tuple(sys.modules.items()):
            if module is None or not module_name.startswith("woof."):
                continue
            for name, value in tuple(vars(module).items()):
                original = reverse.get(id(value))
                if original is not None:
                    setattr(module, name, original)
                    scope.aliases_restored.append(f"{module_name}.{name}")
        for module, name, original in reversed(changes):
            setattr(module, name, original)
        scope.restored = True
