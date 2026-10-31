"""Content-keyed cache for deterministic host preparation.

THE COST THIS REMOVES, MEASURED (2026-10-05, RTX PRO 6000 Blackwell,
published x1.655362 at 30 km, cold start): the forecast door spent 633 s
between the command and the end of the first forecast step, and most of it
was host preparation that recomputes the same arrays from the same bytes on
every run of the same mesh:

* ``_validate_v841_host_mesh``              91.9 s
* ``initialize_deformation_weights_v841``   84.9 s
* ``build_v841_vertical_velocity_damping``  52.2 s, computed TWICE per run
  (the prepared-execution fingerprint and the device stack each build it
  from the same ``zgrid``)
* ``build_advection_coefficients``          46.5 s (twice per run on a
  limited-area mesh: host preparation and the regional driver)
* ``_frozen_vertical_damping``              21.7 s

On conus12km (157,859 cells x 59 layers) the same functions cost 57 s of a
109 s warm start.  None of these values depends on the forecast cycle: they
are functions of the mesh, the vertical grid and a handful of configuration
scalars, so a second run of the same mesh -- every cycle of a production
line -- recomputed them for nothing.

WHAT A CACHED RESULT IS.  A decorated function is keyed on the SHA-256 of a
canonical encoding of EVERYTHING it is called with: every array's dtype,
shape and bytes, every scalar, and the whole instance state of every object
argument (a ``Mesh`` contributes all of its arrays, dimensions, attributes and
paths), together with a producer digest over every ``woof.hex`` source file,
the NumPy version, the interpreter and the C library.  An argument this
module cannot encode canonically (a device array, a callable, an object from
outside ``woof.hex``) makes the call uncacheable and the function simply runs.
A hit returns arrays with the dtype, shape, memory order and bytes the
function returned when it ran, so a forecast cannot tell a hit from a
recomputation: byte identity of the forecast is the contract, and the
receipts of record measure it.

Two layers:

* in process, so the second call of a run (the dss pair, the regional
  advection pair) costs a copy instead of a recomputation;
* on disk, under ``$WOOF_HEX_PREP_CACHE`` (default
  ``$XDG_CACHE_HOME/gpuwm-hex/prep``, else ``~/.cache/gpuwm-hex/prep``), so
  the next run of the same mesh starts from the stored arrays.  Entries are
  written to a private temporary directory and published with one atomic
  rename; a concurrent writer of the same key loses the rename and discards
  its copy, because both copies are the same bytes.

``WOOF_HEX_PREP_CACHE=off`` disables both layers.

WHAT IS IN THE KEY BESIDES THE ARGUMENTS.

* The binding of every name a cached function can reach.  The key walks the
  function's code, and the code of every hexcore function, class and module
  attribute it names, transitively, and records what each name is bound to
  right now.  A hook rebound to another hexcore function (``rkind_libm``'s
  ``sinf_rkind`` and ``powf_rkind`` are rebindable by design) changes the
  key; a name rebound to anything outside hexcore and the trusted libraries
  (a ctypes bridge to a reference libimf, a test's monkeypatch) makes the
  call uncacheable, so it simply runs.  THE BREAKAGE THIS PREVENTS (measured
  2026-10-05): with ``sinf_rkind`` rebound, the cached
  ``build_v841_vertical_velocity_damping`` returned the default hook's
  arrays, while the rebound computation gives different bytes.  A transparent
  wrapper (a timer, a tracer) declares itself with
  ``__prep_cache_transparent__ = True`` and is keyed as what it wraps.
* NOT the paths a mesh was read from.  A class may name instance fields that
  only record where its bytes came from (``Mesh.prep_cache_provenance``:
  ``grid_path``, ``static_path``, ``provenance``); those fields are left out,
  because every array read from those files is already hashed.  THE BREAKAGE
  THIS PREVENTS (measured 2026-10-05, conus12km warm): the same mesh
  bytes hard-linked into another directory recomputed three tables (32.9 s
  instead of 0.9 s) and stored duplicate entries, so a cache baked into an
  image only served meshes at the bake box's absolute paths.  If any code the
  walk reaches names one of those fields, the call is keyed on it after all.
* NumPy's CPU dispatch (the baseline and dispatched SIMD targets this host
  enables), in the producer digest.  THE BREAKAGE THIS PREVENTS: a cache
  built on an AVX-512 host handed to a host without it, where ``np.sin``,
  ``np.cos`` and ``np.arctan2`` may run other kernels and give other bytes; it
  misses instead.

SIZE.  The disk layer is capped at ``WOOF_HEX_PREP_CACHE_MAX_GB`` (default
48); after a store that takes it over the cap, the least recently used entries
are removed (never one this process holds).  THE BREAKAGE THIS PREVENTS: one
conus12km run writes 701 MB, the 30 km and 15 km meshes several and tens of
GB per source version, and every source edit orphans the whole stored set;
uncapped, the cache grows until the disk is full (a test host's work disk filled on
2026-10-05).

MEMORY.  A stored result is held in process as its entry on disk, not as a
second copy: a repeated call in the same run reads it back (page cache) and
copies it out.  THE BREAKAGE THIS PREVENTS (measured from the lane's own
/usr/bin/time receipts): 15 km host preparation peaked at 60.3 GB against
45.6 GB without the cache, because every result was held twice.  Only a result
that could not be stored is kept in memory.

THE GUARDS, AND THE BREAKAGE EACH PREVENTS.
* Every stored array carries its SHA-256 and is re-hashed on load; a
  mismatch discards the entry and recomputes.  Prevents: a truncated write or
  a flipped bit on disk feeding wrong coefficients into a forecast that
  would otherwise run to completion and look plausible.
* The producer digest covers every ``woof.hex`` source file.  Prevents: a
  fixed transcription returning the pre-fix arrays from an entry written
  before the fix.
* Results are copied out of the in-process layer.  Prevents: a caller that
  writes into a returned array (the driver pads and overlays prepared
  arrays) corrupting the copy the next caller receives.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import dis
import enum
import functools
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path, PurePath
import platform
import shutil
import sys
import tempfile
import threading
import time
import types
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

import numpy as np
from . import time_ledger

SCHEMA = "gpuwm-hex.prep-cache/v2"
ENVIRONMENT = "WOOF_HEX_PREP_CACHE"
CAP_ENVIRONMENT = "WOOF_HEX_PREP_CACHE_MAX_GB"
DEFAULT_CAP_GB = 48.0
#: Libraries whose own functions a cached function may call by a module-level
#: name (``from numpy import sqrt``): their code is not in the producer
#: digest, but their versions are, and a binding is accepted only while it is
#: the library's own object under its own name.
_TRUSTED_LIBRARIES = frozenset(getattr(sys, "stdlib_module_names", ())) | {
    "numpy", "scipy", "cupy", "cupyx", "netCDF4", "cftime",
}
#: Arrays at least this large are hashed on the worker pool.
_PARALLEL_BYTES = 1 << 20
#: Arrays at least this large are digested as SHA-256 over the SHA-256 of each block of this many bytes (C order),
#: every block of every array on the host threads at once (_array_digests).  THE COST THIS REMOVES (RRFS hrrrv5
#: limited area, mv3km 206,726 cells, V6000, 2026-10-28): one whole-array SHA-256 per thread left the terrain
#: coupling and regional dycore keys at ~1.7 s of start-up, the largest array hashed on one core while the card
#: waited (about 11x on NOAA's conus3km).  Still a SHA-256 content digest of every byte, in a fixed block order.
_BLOCK_BYTES = 32 << 20
_MAX_DEPTH = 12

_lock = threading.Lock()
_memory: dict[str, dict[str, Any]] = {}
_ledger: list[dict[str, Any]] = []


class Uncacheable(Exception):
    """An argument or result has no canonical encoding; the call just runs."""


# ---------------------------------------------------------------------------
# where, and whether
# ---------------------------------------------------------------------------
def cache_root() -> Path | None:
    """The disk layer's directory, or None when the cache is switched off."""

    declared = os.environ.get(ENVIRONMENT)
    if declared is not None and declared.strip().lower() in ("off", "0", "no", "false"):
        return None
    if declared:
        return Path(declared).expanduser()
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".cache"
    return root / "woof hex" / "prep"


def enabled() -> bool:
    return cache_root() is not None


@functools.lru_cache(maxsize=1)
def producer_digest() -> str:
    """Every hexcore source byte, the NumPy build, the interpreter, the libc."""

    package = Path(__file__).resolve().parent
    digest = hashlib.sha256(SCHEMA.encode("ascii"))
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package).as_posix()
        digest.update(b"\0file\0" + relative.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    for item in (
        np.__version__,
        sys.version,
        platform.machine(),
        " ".join(platform.libc_ver()),
        numpy_cpu_dispatch(),
    ):
        digest.update(b"\0env\0" + str(item).encode("utf-8"))
    return digest.hexdigest()


@functools.lru_cache(maxsize=1)
def numpy_cpu_dispatch() -> str:
    """The SIMD targets NumPy runs on this host: its baseline plus every
    dispatched target this CPU enables (``NPY_DISABLE_CPU_FEATURES`` included)."""

    try:
        from numpy._core import _multiarray_umath as umath
    except ImportError:  # NumPy 1.x
        from numpy.core import _multiarray_umath as umath  # type: ignore[no-redef]
    features = dict(getattr(umath, "__cpu_features__", {}) or {})
    baseline = sorted(getattr(umath, "__cpu_baseline__", []) or [])
    dispatch = sorted(getattr(umath, "__cpu_dispatch__", []) or [])
    enabled = sorted(name for name in set(baseline) | set(dispatch) if features.get(name))
    return json.dumps({"baseline": baseline, "enabled": enabled}, sort_keys=True)


# ---------------------------------------------------------------------------
# canonical encoding of arguments
# ---------------------------------------------------------------------------
class _Encoder:
    """Walk an argument tree into a token stream; arrays are hashed in parallel."""

    def __init__(self, exclude_provenance: bool = True, keep: frozenset[str] = frozenset()) -> None:
        self.tokens: list[bytes | int] = []
        self.arrays: list[np.ndarray] = []
        self._seen_arrays: dict[int, int] = {}
        self._active: set[int] = set()
        self.exclude_provenance = exclude_provenance
        #: Provenance-only names encoded after all (the reached code reads them; call_key).
        self.keep = keep
        #: Provenance-only fields left out of the key, by name.
        self.skipped: set[str] = set()
        #: hexcore classes of encoded objects; their methods join the binding walk.
        self.classes: dict[int, type] = {}

    def add(self, value: Any, depth: int = 0) -> None:
        if depth > _MAX_DEPTH:
            raise Uncacheable("argument nesting deeper than the encoder walks")
        out = self.tokens.append
        if value is None:
            out(b"N")
        elif isinstance(value, bool):
            out(b"B1" if value else b"B0")
        elif isinstance(value, enum.Enum):
            out(b"E" + _qualname(type(value)).encode() + b"\0" + repr(value.value).encode())
        elif isinstance(value, int):
            out(b"I" + str(value).encode("ascii"))
        elif isinstance(value, float):
            out(b"F" + value.hex().encode("ascii"))
        elif isinstance(value, str):
            raw = value.encode("utf-8")
            out(b"S%d:" % len(raw) + raw)
        elif isinstance(value, bytes):
            out(b"Y%d:" % len(value) + value)
        elif isinstance(value, PurePath):
            raw = str(value).encode("utf-8")
            out(b"P%d:" % len(raw) + raw)
        elif isinstance(value, np.generic):
            if value.dtype.hasobject:
                raise Uncacheable("object-dtype scalar")
            out(b"G" + value.dtype.str.encode() + b":" + value.tobytes())
        elif type(value) is np.ndarray:
            self._array(value)
        elif isinstance(value, (tuple, list)):
            out((b"T%d:" if isinstance(value, tuple) else b"L%d:") % len(value))
            for item in value:
                self.add(item, depth + 1)
        elif isinstance(value, (dict, MappingProxyType)):
            keys = list(value)
            if not all(isinstance(key, str) for key in keys):
                raise Uncacheable("mapping with non-string keys")
            out(b"D%d:" % len(keys))
            for key in sorted(keys):
                self.add(key, depth + 1)
                self.add(value[key], depth + 1)
        elif isinstance(value, (set, frozenset)):
            items = list(value)
            if not all(isinstance(item, (str, int)) and not isinstance(item, bool) for item in items):
                raise Uncacheable("set of non-primitive members")
            out(b"Z%d:" % len(items))
            for item in sorted(items, key=lambda item: (isinstance(item, str), item)):
                self.add(item, depth + 1)
        elif isinstance(value, type):
            if not value.__module__.startswith("woof.hex") and value.__module__ not in ("builtins", "numpy"):
                raise Uncacheable(f"class argument {_qualname(value)}")
            out(b"C" + _qualname(value).encode())
        elif isinstance(value, np.dtype):
            out(b"t" + value.str.encode())
        else:
            self._object(value, depth)

    def _array(self, array: np.ndarray) -> None:
        if array.dtype.hasobject:
            raise Uncacheable("object-dtype array")
        index = self._seen_arrays.get(id(array))
        if index is None:
            index = len(self.arrays)
            self.arrays.append(array)
            self._seen_arrays[id(array)] = index
        self.tokens.append(
            b"A" + array.dtype.str.encode() + b":" + repr(array.shape).encode()
        )
        self.tokens.append(index)

    def _object(self, value: Any, depth: int) -> None:
        cls = type(value)
        module = getattr(cls, "__module__", "")
        if not module.startswith("woof.hex"):
            raise Uncacheable(f"argument of foreign type {_qualname(cls)}")
        marker = id(value)
        if marker in self._active:
            raise Uncacheable("cyclic object argument")
        self._active.add(marker)
        try:
            state: dict[str, Any] = {}
            if hasattr(value, "__dict__"):
                state.update(vars(value))
            for slot in _slots(cls):
                if hasattr(value, slot):
                    state[slot] = getattr(value, slot)
            self.classes[id(cls)] = cls
            if self.exclude_provenance:
                for name in getattr(cls, "prep_cache_provenance", ()):
                    if name in state and name not in self.keep:
                        del state[name]
                        self.skipped.add(name)
            self.tokens.append(b"O" + _qualname(cls).encode())
            self.add(state, depth + 1)
        finally:
            self._active.discard(marker)


# ---------------------------------------------------------------------------
# the bindings a cached function reaches
# ---------------------------------------------------------------------------
def _is_hexcore(module_name: Any) -> bool:
    return isinstance(module_name, str) and (module_name == "woof.hex" or module_name.startswith("woof.hex."))


def _trusted(module_name: Any) -> bool:
    return isinstance(module_name, str) and module_name.split(".")[0] in _TRUSTED_LIBRARIES


def _canonical(value: Any, module_name: Any, name: Any) -> bool:
    """True while ``value`` is its home module's own object under its own name."""

    if not isinstance(module_name, str) or not isinstance(name, str):
        return False
    module = sys.modules.get(module_name)
    return module is not None and getattr(module, name, None) is value


# The three code walks below are pure functions of the code object (and the
# package that resolves relative imports), so each is computed once per process.
# THE BREAKAGE THIS PREVENTS, MEASURED (unified image, 4x RTX PRO 6000 box,
# 2026-10-16, cProfile of one 3-member 120 km group): dis.get_instructions over
# the same driver code ran for every member and every keyed preparation, about
# 2.5 s of one core per forecast process with the card idle.  Frozen results
# keep a caller from editing a cached value.
@functools.lru_cache(maxsize=None)
def _code_names(code: types.CodeType) -> frozenset[str]:
    names = set(code.co_names)
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            names |= _code_names(constant)
    return frozenset(names)


@functools.lru_cache(maxsize=None)
def _local_hexcore_imports(code: types.CodeType, package: str) -> frozenset[str]:
    """The hexcore modules ``code`` (nested code included) imports inside a function body.

    THE BREAKAGE THIS PREVENTS (measured 2026-10-12, RTX PRO 6000 host): the walk found a
    function-local ``from .rust_hostprep import vertical_velocity_damping``
    only when ``woof.hex.rust_hostprep`` already sat in ``sys.modules``, so the
    key of ``v841-vertical-velocity-damping`` had 3 binding tokens in a fresh
    process and 25 after any other code had imported that module: one
    computation, two keys, and a rebinding of the Rust table function went
    unseen until the import happened to have run.
    """

    found: set[str] = set()
    instructions = list(dis.get_instructions(code))
    for index, instruction in enumerate(instructions):
        if instruction.opname != "IMPORT_NAME":
            continue
        level = instructions[index - 2].argval if index >= 2 else 0
        fromlist = instructions[index - 1].argval if index >= 1 else None
        level = level if isinstance(level, int) else 0
        name = str(instruction.argval or "")
        try:
            base = importlib.util.resolve_name("." * level + name, package) if level else name
        except (ImportError, ValueError):
            continue
        if not base or not _is_hexcore(base):
            continue
        candidates = [base]
        if isinstance(fromlist, tuple):
            candidates += [f"{base}.{entry}" for entry in fromlist if isinstance(entry, str)]
        for candidate in candidates:
            try:
                spec = importlib.util.find_spec(candidate)
            except (ImportError, ValueError):
                spec = None
            if spec is not None:
                found.add(candidate)
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            found |= _local_hexcore_imports(constant, package)
    return frozenset(found)


@functools.lru_cache(maxsize=None)
def _code_reads(code: types.CodeType) -> frozenset[str]:
    """Every name ``code`` (nested code included) can READ: attribute and
    global loads, and every string constant (``getattr(mesh, "grid_path")``,
    ``vars(mesh)["grid_path"]``).  Stores are not reads."""

    reads: set[str] = set()
    for instruction in dis.get_instructions(code):
        if instruction.opname.startswith(("LOAD_ATTR", "LOAD_METHOD", "LOAD_SUPER_ATTR",
                                          "LOAD_GLOBAL", "LOAD_NAME", "LOAD_FROM_DICT")):
            if isinstance(instruction.argval, str):
                reads.add(instruction.argval)
    for constant in code.co_consts:
        if isinstance(constant, str):
            reads.add(constant)
        elif isinstance(constant, types.CodeType):
            reads |= _code_reads(constant)
    return frozenset(reads)


class _BindingWalk:
    """Every name a function can reach through hexcore code, and its binding."""

    def __init__(self) -> None:
        self.tokens: set[str] = set()
        self.names: set[str] = set()
        #: Names the reached code reads, outside the methods of the class
        #: that declares them provenance-only (its constructor records them,
        #: its copy passes them on; neither is a computation reading a path).
        self.reads: set[str] = set()
        self._codes: set[int] = set()
        self._classes: set[int] = set()
        self._queue: list[tuple[types.CodeType, dict[str, Any], tuple | None, frozenset[str]]] = []
        #: Methods of reached classes, walked once reached code names them
        #: (attribute, global or string constant) or when they are dunders,
        #: which the interpreter calls implicitly.  A class's reader that no
        #: reached code calls (Mesh.from_netcdf, and the netCDF4 binding it
        #: uses) is not part of what the cached function computes.
        self._members: list[tuple[str, str, Any, frozenset[str]]] = []
        self._member_names: set[str] = set()

    def run(self, root: Callable[..., Any], classes: Any) -> None:
        root = self._unwrap(root)
        if isinstance(root, types.FunctionType):
            self._enqueue(root)
        for cls in classes:
            self._visit_class(f"argument:{_qualname(cls)}", cls)
        while True:
            self._drain()
            due = [row for row in self._members
                   if (row[1].startswith("__") and row[1].endswith("__")) or row[1] in self._member_names]
            if not due:
                return
            self._members = [row for row in self._members if not any(row is d for d in due)]
            for owner, attr, member, exempt in due:
                self._visit(owner, attr, member, exempt)

    def _drain(self) -> None:
        while self._queue:
            code, namespace, closure, exempt = self._queue.pop()
            names = _code_names(code)
            reads = _code_reads(code)
            self.names |= names
            self.reads |= reads - exempt
            self._member_names |= names | reads
            owner = str(namespace.get("__name__", "?"))
            package = str(namespace.get("__package__") or "")
            # A function-local import binds at call time; import it now so the
            # key does not depend on whether other code imported it first.
            for module_name in sorted(_local_hexcore_imports(code, package)):
                if module_name not in sys.modules:
                    try:
                        importlib.import_module(module_name)
                    except ImportError:
                        continue    # the call itself would fail the same way
            for name in sorted(names):
                if name in namespace:
                    value = namespace[name]
                    if isinstance(value, types.ModuleType):
                        self._resolve_module(value, names)
                    else:
                        self._visit(owner, name, value)
                for candidate in (name, f"{package}.{name}" if package else None):
                    module = sys.modules.get(candidate) if candidate else None
                    if module is not None and _is_hexcore(module.__name__):
                        self._resolve_module(module, names)  # a function-local import
            for index, cell in enumerate(closure or ()):
                try:
                    content = cell.cell_contents
                except ValueError:  # an empty cell
                    continue
                if isinstance(content, types.ModuleType):
                    self._resolve_module(content, names)
                elif callable(content):
                    self._visit(f"{owner}:{getattr(code, 'co_qualname', code.co_name)}",
                                f"<closure {code.co_freevars[index]}>", content)

    def _resolve_module(self, module: types.ModuleType, names: set[str], depth: int = 0) -> None:
        if not _is_hexcore(module.__name__) or depth > 4:
            return
        namespace = vars(module)
        for name in sorted(names):
            if name not in namespace:
                continue
            value = namespace[name]
            if isinstance(value, types.ModuleType):
                self._resolve_module(value, names, depth + 1)
            else:
                self._visit(module.__name__, name, value)

    @staticmethod
    def _unwrap(value: Any) -> Any:
        for _ in range(16):
            if getattr(value, "__prep_cache_transparent__", False) and hasattr(value, "__wrapped__"):
                value = value.__wrapped__
            else:
                break
        return value

    def _enqueue(self, function: types.FunctionType, exempt: frozenset[str] = frozenset()) -> None:
        code = function.__code__
        if id(code) in self._codes:
            return
        self._codes.add(id(code))
        self._queue.append((code, function.__globals__, function.__closure__, exempt))

    def _visit(self, owner: str, name: str, value: Any, exempt: frozenset[str] = frozenset()) -> None:
        value = self._unwrap(value)
        binding = f"{owner}.{name}"
        if owner == "woof.hex.native_geometry_math:_intrinsic.<locals>.evaluate" and name == "<closure native>":
            from .native_geometry_math import cache_binding
            native_binding = cache_binding(value)
            if native_binding is not None:
                self.tokens.add(f"{binding}={native_binding}")
                return
        if isinstance(value, types.FunctionType):
            home = value.__globals__.get("__name__", "")
            identity = f"{home}:{value.__qualname__}"
            if _is_hexcore(home):
                self.tokens.add(f"{binding}={identity}")
                self._enqueue(value, exempt)
                return
            if _trusted(home) and _canonical(value, value.__module__, value.__name__):
                self.tokens.add(f"{binding}={identity}")
                return
            if _trusted(home) and hasattr(value, "__wrapped__"):
                # A trusted library's decorator around other code (the
                # reprlib.recursive_repr a dataclass puts on __repr__): keyed
                # as the decorator, and what it wraps is walked.
                self.tokens.add(f"{binding}=wrapper:{identity}")
                self._visit(owner, f"{name}<wrapped>", value.__wrapped__, exempt)
                return
            raise Uncacheable(f"{binding} is bound to {identity}, which is outside hexcore")
        if isinstance(value, type):
            self.tokens.add(f"{binding}=class:{_qualname(value)}")
            self._visit_class(binding, value)
            return
        if isinstance(value, (staticmethod, classmethod)):
            self._visit(owner, name, value.__func__, exempt)
            return
        if isinstance(value, property):
            for part, accessor in (("get", value.fget), ("set", value.fset)):
                if accessor is not None:
                    self._visit(owner, f"{name}<{part}>", accessor, exempt)
            return
        if not callable(value):
            return
        if hasattr(value, "__wrapped__") and _trusted(type(value).__module__):
            # functools.lru_cache and the like; a wrapper object from outside
            # the trusted libraries could answer differently from what it
            # wraps, so it falls through to the refusal below.
            self.tokens.add(f"{binding}=wrapper:{_qualname(type(value))}")
            self._visit(owner, f"{name}<wrapped>", value.__wrapped__)
            return
        kind = type(value)
        module_name = getattr(value, "__module__", None) or kind.__module__
        if _trusted(module_name) and _canonical(value, module_name, getattr(value, "__name__", None)):
            self.tokens.add(f"{binding}={module_name}:{value.__name__}")
            return
        if kind.__module__ in ("typing", "types"):
            return  # an annotation object (a generic alias), never called for a value
        if _is_hexcore(kind.__module__):
            self.tokens.add(f"{binding}=instance:{_qualname(kind)}")
            self._visit_class(binding, kind)
            return
        raise Uncacheable(f"{binding} is bound to a {_qualname(kind)}, which is outside hexcore")

    def _visit_class(self, binding: str, cls: type) -> None:
        if not _is_hexcore(cls.__module__) or id(cls) in self._classes:
            return
        self._classes.add(id(cls))
        for klass in cls.__mro__:
            if not _is_hexcore(klass.__module__):
                continue
            exempt = frozenset(getattr(klass, "prep_cache_provenance", ()))
            for attr, member in sorted(vars(klass).items()):
                if isinstance(member, (types.FunctionType, staticmethod, classmethod, property)):
                    self._members.append((_qualname(klass), attr, member, exempt))


def binding_tokens(function: Callable[..., Any], classes: Any = ()) -> tuple[list[str], set[str]]:
    """The bindings ``function`` reaches (sorted, for the key) and every name its code reads
    (outside the methods of a class that declares a name provenance-only).

    Raises :class:`Uncacheable` when a reached name is bound outside hexcore
    and the trusted libraries.
    """

    walk = _BindingWalk()
    walk.run(function, list(classes))
    return sorted(walk.tokens), walk.reads


def _slots(cls: type) -> list[str]:
    names: list[str] = []
    for klass in cls.__mro__:
        slots = getattr(klass, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        names.extend(name for name in slots if name not in ("__dict__", "__weakref__"))
    return names


def _qualname(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _array_bytes(array: np.ndarray) -> memoryview:
    """The array's C-order bytes (``np.ascontiguousarray``; large strided copies in blocks on host threads)."""
    if array.nbytes >= _PARALLEL_BYTES:
        from .host_parallel import contiguous_copy
        return memoryview(contiguous_copy(array, dtype=array.dtype)).cast("B")
    return memoryview(np.ascontiguousarray(array)).cast("B")


def _block_slices(nbytes: int) -> list[slice]:
    return [slice(start, min(start + _BLOCK_BYTES, nbytes)) for start in range(0, nbytes, _BLOCK_BYTES)]


def _combine_blocks(blocks: list[bytes]) -> bytes:
    return hashlib.sha256(b"sha256-blocks\0%d\0" % _BLOCK_BYTES + b"".join(blocks)).digest()


def _array_digest(array: np.ndarray) -> bytes:
    """One array's content digest, on the calling thread (the definition ``_array_digests`` computes in blocks)."""
    data = _array_bytes(array)
    if data.nbytes < _BLOCK_BYTES:
        return hashlib.sha256(data).digest()
    return _combine_blocks([hashlib.sha256(data[block]).digest() for block in _block_slices(data.nbytes)])


def _array_digests(arrays: list[np.ndarray]) -> list[bytes]:
    """``[_array_digest(a) for a in arrays]``, every block of every large array hashed on host threads at once
    (hashlib releases the GIL), so one big array no longer runs on one core while the others finish."""
    from .host_parallel import threads as host_threads
    digests: list[bytes | None] = [None] * len(arrays)
    jobs: list[tuple[int, memoryview]] = []
    blocked: dict[int, int] = {}
    for i, array in enumerate(arrays):
        if array.nbytes < _PARALLEL_BYTES:
            digests[i] = _array_digest(array)
            continue
        data = _array_bytes(array)
        if data.nbytes < _BLOCK_BYTES:
            jobs.append((i, data))
            continue
        blocked[i] = len(jobs)
        jobs.extend((i, data[block]) for block in _block_slices(data.nbytes))
    workers = min(len(jobs), max(1, host_threads()))
    if workers > 1:
        with concurrent.futures.ThreadPoolExecutor(workers, thread_name_prefix="hex-prep-digest") as pool:
            results = list(pool.map(lambda job: hashlib.sha256(job[1]).digest(), jobs))
    else:
        results = [hashlib.sha256(job[1]).digest() for job in jobs]
    for i, start in blocked.items():
        count = len(_block_slices(arrays[i].nbytes))
        digests[i] = _combine_blocks(results[start:start + count])
    for (i, _), value in zip(jobs, results):
        if digests[i] is None:
            digests[i] = value
    return digests  # type: ignore[return-value]


def _encode_arguments(args: tuple, kwargs: Mapping[str, Any], exclude_provenance: bool,
                      keep: frozenset[str] = frozenset()) -> _Encoder:
    encoder = _Encoder(exclude_provenance, keep)
    encoder.add(tuple(args))
    encoder.add(dict(kwargs))
    return encoder


def call_key(label: str, function: Callable[..., Any], args: tuple, kwargs: Mapping[str, Any],
             salt: str = "", independent_of: Iterable[str] = ()) -> str:
    encoder = _encode_arguments(args, kwargs, True)
    bindings, reads = binding_tokens(function, encoder.classes.values())
    needed = frozenset(encoder.skipped & reads) - frozenset(independent_of)
    if needed:
        # Reached code names a provenance field: key on that field after all, and only on that one.  THE COST
        # THIS REMOVES (RRFS hrrrv5 limited area, same-host rank processes, mv3km 4 ranks on V8X6000,
        # 2026-10-29): one read name re-encoded every provenance field, so a rank mesh's call was keyed on the
        # whole mesh it was cut from (``partition_source_mesh``): every rank process hashed all of it (2.0 GB on
        # mv3km, 23.9 GB on NOAA's conus3km) and held it resident.
        encoder = _encode_arguments(args, kwargs, True, needed)
    digests = _array_digests(encoder.arrays)
    key = hashlib.sha256()
    for part in (SCHEMA, producer_digest(), label, f"{function.__module__}.{function.__qualname__}", salt):
        key.update(part.encode("utf-8") + b"\0")
    for token in bindings:
        key.update(b"bind|" + token.encode("utf-8") + b"\0")
    for token in encoder.tokens:
        if isinstance(token, int):
            key.update(b"#" + digests[token])
        else:
            key.update(b"%d|" % len(token) + token)
    return key.hexdigest()


# ---------------------------------------------------------------------------
# results: exact round trip of arrays, scalars, containers and dataclasses
# ---------------------------------------------------------------------------
def _encode_result(value: Any, arrays: list[np.ndarray], depth: int = 0) -> Any:
    if depth > _MAX_DEPTH:
        raise Uncacheable("result nesting deeper than the encoder walks")
    if value is None or isinstance(value, (bool, str)):
        return {"v": value}
    if isinstance(value, int):
        return {"i": str(value)}
    if isinstance(value, float):
        return {"f": value.hex()}
    if isinstance(value, np.generic):
        if value.dtype.hasobject:
            raise Uncacheable("object-dtype scalar result")
        arrays.append(np.asarray(value))
        return {"g": len(arrays) - 1}
    if type(value) is np.ndarray:
        if value.dtype.hasobject:
            raise Uncacheable("object-dtype array result")
        arrays.append(value)
        return {
            "a": len(arrays) - 1,
            "order": "F" if (value.flags.f_contiguous and not value.flags.c_contiguous) else "C",
            "writeable": bool(value.flags.writeable),
        }
    if isinstance(value, (tuple, list)):
        tag = "t" if isinstance(value, tuple) else "l"
        return {tag: [_encode_result(item, arrays, depth + 1) for item in value]}
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {"d": {k: _encode_result(v, arrays, depth + 1) for k, v in value.items()}}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        cls = type(value)
        if not cls.__module__.startswith("woof.hex"):
            raise Uncacheable(f"dataclass result of foreign type {_qualname(cls)}")
        fields = dataclasses.fields(value)
        if any(not field.init for field in fields):
            raise Uncacheable(f"dataclass {_qualname(cls)} has init=False fields")
        extra = set(getattr(value, "__dict__", {})) - {field.name for field in fields}
        if extra:
            raise Uncacheable(f"dataclass {_qualname(cls)} carries undeclared state {sorted(extra)}")
        return {
            "dc": [cls.__module__, cls.__qualname__],
            "fields": {field.name: _encode_result(getattr(value, field.name), arrays, depth + 1) for field in fields},
        }
    raise Uncacheable(f"result of type {_qualname(type(value))}")


def _decode_result(node: Mapping[str, Any], arrays: list[np.ndarray]) -> Any:
    if "v" in node:
        return node["v"]
    if "i" in node:
        return int(node["i"])
    if "f" in node:
        return float.fromhex(node["f"])
    if "g" in node:
        return arrays[node["g"]][()]
    if "a" in node:
        source = arrays[node["a"]]
        array = np.array(source, copy=True, order=node["order"])
        if not node["writeable"]:
            array.flags.writeable = False
        return array
    if "t" in node:
        return tuple(_decode_result(item, arrays) for item in node["t"])
    if "l" in node:
        return [_decode_result(item, arrays) for item in node["l"]]
    if "d" in node:
        return {k: _decode_result(v, arrays) for k, v in node["d"].items()}
    if "dc" in node:
        module_name, qualname = node["dc"]
        target: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            target = getattr(target, part)
        return target(**{k: _decode_result(v, arrays) for k, v in node["fields"].items()})
    raise ValueError(f"unknown prep-cache node {sorted(node)}")


# ---------------------------------------------------------------------------
# the disk layer
# ---------------------------------------------------------------------------
def _entry_dir(root: Path, key: str) -> Path:
    return root / key[:2] / key


def _store(root: Path, key: str, label: str, structure: Any, arrays: list[np.ndarray], digests: list[str]) -> None:
    final = _entry_dir(root, key)
    if final.exists():
        return
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{key[:12]}.", dir=final.parent))
    try:
        for index, array in enumerate(arrays):
            np.save(staging / f"{index}.npy", array, allow_pickle=False)
        manifest = {
            "schema": SCHEMA,
            "key": key,
            "label": label,
            "producer_digest": producer_digest(),
            "structure": structure,
            "arrays": [
                {"dtype": a.dtype.str, "shape": list(a.shape), "sha256": d}
                for a, d in zip(arrays, digests)
            ],
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        try:
            os.rename(staging, final)
        except OSError:
            pass  # another writer published the same key first; same bytes
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _load(root: Path, key: str) -> tuple[Any, list[np.ndarray]] | None:
    time_ledger.emit("prep.cache_load", "begin", attrs={"key": key})
    try:
        return _load_payload(root, key)
    finally:
        time_ledger.emit("prep.cache_load", "end", attrs={"key": key})


def _load_payload(root: Path, key: str) -> tuple[Any, list[np.ndarray]] | None:
    entry = _entry_dir(root, key)
    manifest_path = entry / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA or manifest.get("key") != key:
            raise ValueError("manifest does not describe this key")
        rows = manifest["arrays"]
        arrays = [_map(entry / f"{index}.npy") for index in range(len(rows))]
        digests = _array_digests(arrays)
        for array, row, digest in zip(arrays, rows, digests):
            if array.dtype.str != row["dtype"] or list(array.shape) != row["shape"] or digest.hex() != row["sha256"]:
                raise ValueError("stored array does not match its recorded digest")
    except Exception as error:  # a damaged entry is discarded and recomputed
        _note({"event": "discarded", "key": key, "reason": f"{type(error).__name__}: {error}"})
        shutil.rmtree(entry, ignore_errors=True)
        return None
    try:
        os.utime(manifest_path)  # least-recently-used order for the size cap
    except OSError:
        pass  # a read-only (baked) cache is still served
    return manifest["structure"], arrays


def _map(path: Path) -> np.ndarray:
    """A stored array, memory-mapped read-only: file pages, not a second heap copy."""

    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except ValueError:  # a zero-size array cannot be mapped
        return np.load(path, allow_pickle=False)


def cap_bytes() -> int | None:
    """The disk layer's size cap in bytes, or None when uncapped (``0`` or ``off``)."""

    declared = os.environ.get(CAP_ENVIRONMENT, "").strip().lower()
    if declared in ("off", "none"):
        return None
    try:
        gigabytes = float(declared) if declared else DEFAULT_CAP_GB
    except ValueError:
        gigabytes = DEFAULT_CAP_GB
    return None if gigabytes <= 0 else int(gigabytes * (1 << 30))


def _entries(root: Path) -> list[tuple[float, int, Path]]:
    """(last use, bytes, directory) for every published entry under ``root``."""

    rows = []
    for shard in root.iterdir() if root.is_dir() else ():
        if not (shard.is_dir() and len(shard.name) == 2):
            continue  # the kernel manifests live in kernels/
        for entry in shard.iterdir():
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            try:
                size = sum(item.stat().st_size for item in entry.iterdir())
                used = (entry / "manifest.json").stat().st_mtime
            except OSError:
                continue
            rows.append((used, size, entry))
    return rows


def evict(root: Path, protect: set[str] | frozenset[str] = frozenset()) -> list[str]:
    """Remove least-recently-used entries until the disk layer fits its cap."""

    cap = cap_bytes()
    if cap is None:
        return []
    rows = _entries(root)
    total = sum(size for _, size, _ in rows)
    removed: list[str] = []
    for used, size, entry in sorted(rows, key=lambda row: row[0]):
        if total <= cap:
            break
        if entry.name in protect:
            continue
        shutil.rmtree(entry, ignore_errors=True)
        total -= size
        removed.append(entry.name)
    if removed:
        _note({"event": "evicted", "entries": len(removed), "remaining_bytes": total, "cap_bytes": cap})
    return removed


# ---------------------------------------------------------------------------
# the decorator
# ---------------------------------------------------------------------------
def _note(row: dict[str, Any]) -> None:
    with _lock:
        _ledger.append(row)


def receipt() -> dict[str, Any]:
    """What the cache did in this process, for the run receipt."""

    with _lock:
        rows = [dict(row) for row in _ledger]
    root = cache_root()
    return {
        "schema": SCHEMA,
        "enabled": root is not None,
        "root": None if root is None else str(root),
        "producer_digest": producer_digest() if root is not None else None,
        "numpy_cpu_dispatch": numpy_cpu_dispatch() if root is not None else None,
        "cap_bytes": cap_bytes() if root is not None else None,
        "calls": rows,
    }


def cached(label: str, salt: Callable[[], str] | None = None,
           independent_of: Iterable[str] = ()) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Serve a deterministic host-preparation function from the content-keyed cache.

    ``salt`` names code the producer digest cannot see -- a Rust bridge the
    function streams its arrays through -- by returning that executable's
    digest; it is folded into every key, so a rebuilt bridge never replays
    the previous build's arrays.  A salt that cannot be computed makes the
    call uncacheable, and the function then raises its own refusal.

    ``independent_of`` names provenance-only fields the function reads but
    whose content its result cannot depend on (the function says why where it
    is declared); they stay out of the key even though the code names them.
    """
    independent = frozenset(independent_of)

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            root = cache_root()
            if root is None:
                return function(*args, **kwargs)
            started = time.perf_counter()
            try:
                extra = "" if salt is None else str(salt())
            except Exception as reason:  # the function names the missing bridge itself
                _note({"label": label, "event": "uncacheable", "reason": f"salt: {reason}"})
                return function(*args, **kwargs)
            try:
                key = call_key(label, function, args, kwargs, extra, independent)
            except Uncacheable as reason:
                _note({"label": label, "event": "uncacheable", "reason": str(reason)})
                return function(*args, **kwargs)
            keyed = time.perf_counter() - started
            with _lock:
                held = _memory.get(key)
            source = "memory"
            if held is not None and "arrays" not in held:
                # Held as its disk entry: read it back (page cache), re-verified.
                loaded = _load(root, key)
                held = None if loaded is None else {"structure": loaded[0], "arrays": loaded[1]}
            if held is None:
                loaded = _load(root, key)
                if loaded is not None:
                    held = {"structure": loaded[0], "arrays": loaded[1]}
                    with _lock:
                        _memory[key] = {"structure": loaded[0], "disk": True}
                    source = "disk"
            if held is not None:
                value = _decode_result(held["structure"], held["arrays"])
                del held
                _note({"label": label, "event": f"hit-{source}", "key": key,
                       "key_seconds": round(keyed, 3),
                       "seconds": round(time.perf_counter() - started, 3)})
                return value
            computed = time.perf_counter()
            result = function(*args, **kwargs)
            compute_seconds = time.perf_counter() - computed
            try:
                arrays: list[np.ndarray] = []
                structure = _encode_result(result, arrays)
            except Uncacheable as reason:
                _note({"label": label, "event": "uncacheable-result", "reason": str(reason)})
                return result
            # Stored before the caller holds the result, so nothing has
            # written into these arrays yet; held in process as the entry.
            try:
                digests = [d.hex() for d in _array_digests(arrays)]
                _store(root, key, label, structure, arrays, digests)
                stored = True
                with _lock:
                    _memory[key] = {"structure": structure, "disk": True}
                    protect = set(_memory)
                if (os.environ.get('WOOF_HEX_MOVING_RESIDENT') == '1'
                        and os.environ.get('WOOF_HEX_MOVING_CACHE_MAINTENANCE','1') != '0'):
                    from .moving_prep_maintenance import request
                    request(root, protect)
                else:
                    evict(root, protect)
            except OSError as error:
                stored = False
                kept = [np.array(array, copy=True, order="K") for array in arrays]
                with _lock:
                    _memory[key] = {"structure": structure, "arrays": kept}
                _note({"label": label, "event": "store-failed", "reason": str(error)})
            _note({"label": label, "event": "miss", "key": key, "stored": stored,
                   "key_seconds": round(keyed, 3), "compute_seconds": round(compute_seconds, 3),
                   "seconds": round(time.perf_counter() - started, 3)})
            return result

        wrapper.__wrapped_prep_cache__ = function  # type: ignore[attr-defined]
        return wrapper

    return decorate
