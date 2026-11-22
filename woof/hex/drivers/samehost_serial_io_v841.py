"""Keep legacy NetCDF extension objects on the forecast coordinator thread.

An explicitly selected free-threaded runtime may retain the unchanged legacy
reader and writer when every covered extension call stays on this one thread.
Guarded objects and conversion entrypoints refuse worker access before calls.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import functools
import sys
import threading


class SerialExtensionIoError(RuntimeError):
    """A worker attempted an extension call that requires serialization."""


class _Guard:
    def __init__(self, raw_types):
        self.coordinator = threading.get_ident()
        self.raw_types = raw_types
        self.calls = {}
        self.worker_refusals = 0
        self.restored = False
        self.aliases_patched = []
        self.new_aliases_restored = []
        self._lock = threading.Lock()

    def check(self, operation):
        if threading.get_ident() != self.coordinator:
            with self._lock:
                self.worker_refusals += 1
            raise SerialExtensionIoError(
                f"{operation}: NetCDF/cftime extension access must remain on the forecast coordinator; "
                "concurrent calls into the legacy native I/O library are not supported"
            )
        self.calls[operation] = self.calls.get(operation, 0) + 1

    def wrap(self, value):
        if isinstance(value, self.raw_types):
            return _ExtensionProxy(self, value)
        if isinstance(value, Mapping):
            return _MappingProxy(self, value)
        if isinstance(value, tuple):
            return tuple(self.wrap(item) for item in value)
        if isinstance(value, list):
            return [self.wrap(item) for item in value]
        return value

    def receipt(self):
        with self._lock:
            refused = self.worker_refusals
        return {"schema": "mpas-port.samehost-serial-extension-io/v1",
                "policy": "coordinator-only legacy NetCDF object and date-conversion access",
                "coordinator_thread_id": self.coordinator,
                "coordinator_calls": dict(self.calls), "worker_refusals": refused,
                "module_entrypoints_restored": self.restored,
                "existing_reader_aliases_patched": list(self.aliases_patched),
                "new_reader_aliases_restored": list(self.new_aliases_restored),
                "covered_objects": ["Dataset", "Group", "Variable", "Dimension"],
                "covered_module_entrypoints": ["netCDF4.Dataset", "netCDF4.MFDataset",
                                                "netCDF4/cftime num2date/date2num/date2index/num2pydate when present"],
                "library_source": "https://github.com/Unidata/netcdf4-python",
                "runtime_policy": "the caller explicitly chooses CPython -X gil=0; the guard changes no interpreter setting",
                "nonclaims": ["thread safety of arbitrary netCDF4/cftime application use",
                              "direct cftime date constructors or methods outside the current global forecast route",
                              "regional LBC I/O concurrency qualification"]}


class _ExtensionProxy:
    __slots__ = ("_guard", "_value")

    def __init__(self, guard, value):
        object.__setattr__(self, "_guard", guard)
        object.__setattr__(self, "_value", value)

    def _access(self, operation):
        guard = object.__getattribute__(self, "_guard")
        value = object.__getattribute__(self, "_value")
        guard.check(f"{type(value).__name__}.{operation}")
        return guard, value

    def __getattribute__(self, name):
        if name in ("_guard", "_value"):
            guard = object.__getattribute__(self, "_guard")
            guard.check(f"NetCDF proxy.getattr/{name}")
        return object.__getattribute__(self, name)

    def __getattr__(self, name):
        guard, value = self._access(f"getattr/{name}")
        result = getattr(value, name)
        if callable(result):
            def invoke(*args, **kwargs):
                guard.check(f"{type(value).__name__}.{name}")
                return guard.wrap(result(*args, **kwargs))
            return invoke
        return guard.wrap(result)

    def __setattr__(self, name, value):
        _guard, target = self._access(f"setattr/{name}")
        setattr(target, name, value)

    def __delattr__(self, name):
        _guard, target = self._access(f"delattr/{name}")
        delattr(target, name)

    def __getitem__(self, key):
        guard, value = self._access("getitem")
        return guard.wrap(value[key])

    def __setitem__(self, key, item):
        _guard, value = self._access("setitem")
        value[key] = item

    def __len__(self):
        _guard, value = self._access("len")
        return len(value)

    def __iter__(self):
        guard, value = self._access("iter")
        for item in value:
            guard.check(f"{type(value).__name__}.next")
            yield guard.wrap(item)

    def __enter__(self):
        guard, value = self._access("enter")
        return guard.wrap(value.__enter__())

    def __exit__(self, *args):
        _guard, value = self._access("exit")
        return value.__exit__(*args)

    def __repr__(self):
        _guard, value = self._access("repr")
        return repr(value)


class _MappingProxy(Mapping):
    def __init__(self, guard, value):
        self._guard, self._value = guard, value

    def __getitem__(self, key):
        self._guard.check("NetCDF mapping.getitem")
        return self._guard.wrap(self._value[key])

    def __getattribute__(self, name):
        if name in ("_guard", "_value"):
            guard = object.__getattribute__(self, "_guard")
            guard.check(f"NetCDF mapping.getattr/{name}")
        return object.__getattribute__(self, name)

    def __iter__(self):
        self._guard.check("NetCDF mapping.iter")
        for key in self._value:
            self._guard.check("NetCDF mapping.next")
            yield key

    def __len__(self):
        self._guard.check("NetCDF mapping.len")
        return len(self._value)


@contextmanager
def coordinator_only_extension_io(*, netcdf=None, cftime=None):
    """Install before the CLI imports readers that bind module entrypoints.

    Extension objects returned by Dataset and its methods retain the guard.
    Source modules and array bytes are unchanged; all module entrypoints are
    restored even when the caller or a worker refuses the run.
    """
    if netcdf is None:
        import netCDF4 as netcdf
    if cftime is None:
        import cftime
    originals = []
    replacements = {}
    reverse_replacements = {}
    raw_types = tuple(value for name in ("Dataset", "Group", "Variable", "Dimension", "MFDataset")
                      if isinstance(value := getattr(netcdf, name, None), type))
    guard = _Guard(raw_types)

    def patch(module, name):
        original = getattr(module, name, None)
        if original is None:
            return
        if getattr(original, "_samehost_serial_io_guard", False):
            raise SerialExtensionIoError("legacy extension I/O guard is already installed")

        @functools.wraps(original, updated=())
        def invoke(*args, **kwargs):
            guard.check(f"{module.__name__}.{name}")
            return guard.wrap(original(*args, **kwargs))

        invoke._samehost_serial_io_guard = True
        originals.append((module, name, original))
        setattr(module, name, invoke)
        replacements[id(original)] = invoke
        reverse_replacements[id(invoke)] = original

    try:
        for name in ("Dataset", "MFDataset", "num2date", "date2num", "date2index", "num2pydate"):
            patch(netcdf, name)
        for name in ("num2date", "date2num", "date2index", "num2pydate"):
            patch(cftime, name)
        # Readers may already have imported Dataset or a conversion function
        # by value. Replace those known aliases before any worker exists.
        for module_name, module in tuple(sys.modules.items()):
            if module is None or not module_name.startswith(("woof.hex.", "woof.")):
                continue
            for name, value in tuple(vars(module).items()):
                replacement = replacements.get(id(value))
                if replacement is not None:
                    originals.append((module, name, value))
                    setattr(module, name, replacement)
                    guard.aliases_patched.append(f"{module_name}.{name}")
        yield guard
    finally:
        # Modules first imported during the context may bind a guarded
        # entrypoint by value too. Restore those newly created aliases before
        # restoring the original module and pre-existing reader bindings.
        for module_name, module in tuple(sys.modules.items()):
            if module is None or not module_name.startswith(("woof.hex.", "woof.")):
                continue
            for name, value in tuple(vars(module).items()):
                original = reverse_replacements.get(id(value))
                if original is not None:
                    setattr(module, name, original)
                    guard.new_aliases_restored.append(f"{module_name}.{name}")
        for module, name, original in reversed(originals):
            setattr(module, name, original)
        guard.restored = True
