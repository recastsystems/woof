"""Resident point-cull members using the unchanged forecast driver and loop.

One process owns one card. Each member has private proof globals, atmosphere,
physics workspaces, regional scrub observer, graph controller and CUDA stream.
Only compiler modules and content-identical read-only host tables are shared.
The door remains opt-in until the complete C3/C4 speed gate qualifies it.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
from types import FunctionType, ModuleType


class MultiMemberRefusal(ValueError):
    """A member roster cannot safely share this resident stack."""


def private_module(module):
    """Clone run globals without copying or rewriting numerical function code."""
    clone = ModuleType(module.__name__)
    namespace = vars(clone)
    namespace.update(vars(module))
    for name, value in tuple(namespace.items()):
        if isinstance(value, (dict, list, set)):
            namespace[name] = copy.deepcopy(value)
        elif isinstance(value, FunctionType) and value.__globals__ is vars(module):
            function = FunctionType(value.__code__, namespace, value.__name__,
                                    value.__defaults__, value.__closure__)
            function.__kwdefaults__ = value.__kwdefaults__
            function.__dict__.update(value.__dict__)
            function.__module__ = value.__module__
            function.__qualname__ = value.__qualname__
            namespace[name] = function
    return clone


class SharedTables:
    """Content-addressed immutable arrays, with checked shm and disk fallback."""

    def __init__(self, root, *, policy="auto", shm_root=Path("/dev/shm"), creation_manifest=None):
        if policy not in ("auto", "nvme"):
            raise MultiMemberRefusal("shared-buffer policy must be auto or nvme")
        self.root = Path(root)
        self.root.mkdir()
        self.created = [str(self.root)]
        self.creation_manifest = None if creation_manifest is None else Path(creation_manifest)
        if self.creation_manifest is not None:
            with self.creation_manifest.open("x", encoding="utf-8") as stream:
                stream.write(str(self.creation_manifest) + "\n" + str(self.root) + "\n")
            self.created.append(str(self.creation_manifest))
        self.shm_root = Path(shm_root)
        self.policy = policy
        self.shm = None
        self.arrays = {}
        self.maps = []
        self.rows = []
        self.deleted = []

    def _created(self, path):
        self.created.append(str(path))
        if self.creation_manifest is not None:
            with self.creation_manifest.open("a", encoding="utf-8") as stream:
                stream.write(str(path) + "\n")

    @staticmethod
    def available(root):
        try:
            return int(shutil.disk_usage(root).free)
        except OSError:
            return 0

    def intern(self, value):
        import numpy as np
        if not isinstance(value, np.ndarray) or value.dtype.hasobject or not value.size:
            return value
        array = np.ascontiguousarray(value).reshape(value.shape)
        digest = hashlib.sha256(memoryview(array).cast("B")).hexdigest()
        key = (array.dtype.str, array.shape, digest)
        if key in self.arrays:
            return self.arrays[key]
        # Reserve the payload plus the npy header BEFORE touching mapped pages.
        # posix_fallocate catches an shm race as ENOSPC rather than SIGBUS.
        need = int(array.nbytes) + 4096
        free = self.available(self.shm_root)
        use_shm = self.policy == "auto" and free >= need
        reason = None if use_shm else ("forced NVMe" if self.policy == "nvme"
                                      else "available shm smaller than shared buffer")
        if use_shm and self.shm is None:
            self.shm = Path(tempfile.mkdtemp(prefix="hex-members-", dir=self.shm_root))
            self._created(self.shm)
        directory = self.shm if use_shm else self.root
        path = directory / f"{len(self.rows):05d}.npy"
        started = time.perf_counter()
        created = False
        try:
            with path.open("xb") as stream:
                created = True
                self._created(path)
                if hasattr(os, "posix_fallocate"):
                    os.posix_fallocate(stream.fileno(), 0, need)
                np.save(stream, array, allow_pickle=False)
                stream.truncate()
        except OSError as error:
            if created and path.exists():
                size = path.stat().st_size
                path.unlink()
                self.deleted.append({"path": str(path), "bytes": size})
            if not use_shm or error.errno not in (errno.ENOSPC, errno.EDQUOT):
                raise
            reason = "shm reservation exhausted during allocation; NVMe fallback"
            path = self.root / f"{len(self.rows):05d}.npy"
            with path.open("xb") as stream:
                self._created(path)
                np.save(stream, array, allow_pickle=False)
            use_shm = False
        mapped = np.load(path, mmap_mode="r", allow_pickle=False)
        self.maps.append(mapped)
        # The existing prep-cache contract accepts plain ndarrays. Retain
        # the mapping as owner, exposing a read-only ndarray view so that
        # identical tables continue to hit the established preparation cache.
        shared = np.asarray(mapped)
        self.arrays[key] = shared
        self.rows.append({"path": str(path), "sha256": digest,
                          "payload_bytes": int(array.nbytes), "shm_free_before": free,
                          "storage": "shm" if use_shm else "NVMe-backed file",
                          "fallback": reason, "seconds": time.perf_counter() - started})
        return shared

    def immutable(self, value):
        """Copy containers; never freeze member state or constructor arrays."""
        import dataclasses
        import numpy as np
        if isinstance(value, np.ndarray):
            return self.intern(value)
        if isinstance(value, dict):
            return {key: self.immutable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self.immutable(item) for item in value)
        if dataclasses.is_dataclass(value) or type(value).__module__.startswith("woof.hex"):
            if dataclasses.is_dataclass(value):
                items = ((field.name, getattr(value, field.name)) for field in dataclasses.fields(value))
            elif hasattr(value, "__dict__"):
                items = vars(value).items()
            else:
                return value
            if items is not None:
                result = copy.copy(value)
                for name, item in items:
                    object.__setattr__(result, name, self.immutable(item))
                return result
        return value

    def host(self, host):
        import dataclasses
        result = dict(host)
        prepared = host["prepared"]
        fields = ({field.name: getattr(prepared, field.name) for field in dataclasses.fields(prepared)}
                  if dataclasses.is_dataclass(prepared) else dict(vars(prepared)))
        # Explicit audited immutable products. State, saved diagnostics and
        # all constructor values remain member-owned even if bytes coincide.
        for name in ("mesh", "vertical", "reference", "terrain_metrics",
                     "advection_coefficients", "reference_wind_profiles"):
            if name in fields:
                fields[name] = self.immutable(fields[name])
        result["prepared"] = copy.copy(prepared)
        for name, value in fields.items():
            object.__setattr__(result["prepared"], name, value)
        return result

    def receipt(self):
        return {"schema": "gpuwm-hex.shared-member-tables/v1", "policy": self.policy,
                "buffers": list(self.rows), "created": list(self.created), "deleted": list(self.deleted)}

    def close(self):
        for mapped in self.maps:
            mapped._mmap.close()
        self.arrays.clear()
        self.maps.clear()
        for row in self.rows:
            path = Path(row["path"])
            size = path.stat().st_size
            path.unlink()
            self.deleted.append({"path": str(path), "bytes": size})
        if self.shm is not None:
            self.shm.rmdir()
            self.deleted.append({"path": str(self.shm), "bytes": 0, "kind": "directory"})
        self.root.rmdir()
        self.deleted.append({"path": str(self.root), "bytes": 0, "kind": "directory"})


def kernel_cache_family(base):
    """Share compiler ownership, retain each regional post-launch observer."""
    lock = threading.RLock()
    owner = None

    class MemberCache(base):
        def __init__(self, **kwargs):
            nonlocal owner
            with lock:
                if owner is None:
                    owner = base(**kwargs)
                self.__dict__.update(owner.__dict__)
                self._observed = {}
                self.post_launch = None
                self._resolved = {}

        def raw_kernel(self, *args, **kwargs):
            # One worker owns each member cache. Resolve a signature through
            # the compiler once, then reuse that member's handle without
            # taking the family lock or hashing the CUDA source again.
            # Include the observer so installing a regional scrub callback
            # after construction cannot return an earlier bare handle.
            signature = (id(self.post_launch), getattr(self, "base_options", None), args, tuple(
                (key, tuple(value) if key == "options" else value)
                for key, value in sorted(kwargs.items())))
            cached = self._resolved.get(signature)
            if cached is not None:
                return cached
            with lock:
                resolved = super().raw_kernel(*args, **kwargs)
            self._resolved[signature] = resolved
            return resolved

        def compile_manifest(self):
            with lock:
                return super().compile_manifest()

        def clear_memory_cache(self):
            raise MultiMemberRefusal("cannot clear a kernel cache while resident members own its modules")

    MemberCache.family_lock = lock
    return MemberCache


@contextmanager
def capture_exclusion(lock):
    """No device-wide synchronize while another member captures a CUDA graph.

    THE BREAKAGE THIS PREVENTS (B5, 2026-10-16, two GEFS members resident on the 240 km globe, RTX
    5090): one member loaded a kernel for its first step (the cache's load ends in
    ``cudaDeviceSynchronize``) while the other was capturing its subcycle graph; CUDA refuses a device
    synchronize while any stream captures (``cudaErrorStreamCaptureUnsupported``), both members'
    step 0 aborted, and the roster died.  Point-cull rosters had not met it.  Here a capture holds the
    kernel family's lock from ``begin_capture`` to ``end_capture``, and every device synchronize and
    kernel load waits for that lock, so they run before or after a capture, never inside one.  The
    capturing thread itself re-enters the lock, so its own path is unchanged.
    """
    import cupy as cp

    stream_class, runtime = cp.cuda.Stream, cp.cuda.runtime
    begin, end, synchronize = stream_class.begin_capture, stream_class.end_capture, runtime.deviceSynchronize
    held = threading.local()

    def begin_capture(self, *args, **kwargs):
        lock.acquire()
        try:
            value = begin(self, *args, **kwargs)
        except BaseException:
            lock.release()
            raise
        held.depth = getattr(held, "depth", 0) + 1
        return value

    def end_capture(self, *args, **kwargs):
        try:
            return end(self, *args, **kwargs)
        finally:
            if getattr(held, "depth", 0) > 0:
                held.depth -= 1
                lock.release()

    def device_synchronize():
        with lock:
            return synchronize()

    stream_class.begin_capture, stream_class.end_capture = begin_capture, end_capture
    runtime.deviceSynchronize = device_synchronize
    try:
        yield
    finally:
        stream_class.begin_capture, stream_class.end_capture = begin, end
        runtime.deviceSynchronize = synchronize


@contextmanager
def shared_device_tables(*, containers=None):
    """Reuse the static arrays already marked immutable by regional residency."""
    if containers is None:
        from .cuda_backend import containers
    import numpy as np
    mesh_names = ("cellsOnEdge", "edgesOnCell", "nEdgesOnCell", "cellsOnCell",
        "verticesOnEdge", "edgesOnEdge", "nEdgesOnEdge", "verticesOnCell",
        "edgesOnVertex", "cellsOnVertex", "weightsOnEdge", "dcEdge", "dvEdge",
        "areaCell", "areaTriangle", "kiteAreasOnVertex", "latCell", "lonCell",
        "latEdge", "lonEdge", "angleEdge", "meshDensity", "defc_a", "defc_b",
        "fVertex", "fEdge", "spec_zone_mask_edge", "nominalMinDc", "nominal_min_dc")
    specifications = {
        "DeviceMesh": mesh_names,
        "DeviceVerticalGrid": ("zw", "dzw", "rdzw", "zu", "dzu", "rdzu",
            "rdzwp", "rdzwm", "fzp", "fzm", "ah", "hx", "zgrid", "zz",
            "zxu", "dss", "cf1", "cf2", "cf3", "first_height_level"),
        "DeviceReferenceState": ("rho_base", "rho_theta_base", "pressure_base", "exner_base"),
        "DeviceTerrainMetrics": ("zb_cell", "zb3_cell"),
    }
    changes, objects = [], {}
    counters = {name: {"constructed": 0, "reused": 0} for name in specifications}
    lock = threading.RLock()

    def factory(name, original):
        def load(cls, host, **kwargs):
            digest = hashlib.sha256()
            digest.update(repr(sorted((key, str(value)) for key, value in kwargs.items())).encode())
            digest.update(repr(sorted(getattr(host, "dimensions", {}).items())).encode())
            for field in specifications[name]:
                value = containers._mesh_value(host, field) if name == "DeviceMesh" else getattr(host, field)
                if value is None:
                    digest.update(b"none\0")
                else:
                    array = np.ascontiguousarray(value)
                    if array.dtype.hasobject:
                        raise MultiMemberRefusal("immutable device tables cannot contain object arrays")
                    digest.update(array.dtype.str.encode() + repr(array.shape).encode())
                    digest.update(memoryview(array).cast("B"))
            key = (name, digest.hexdigest())
            with lock:
                if key not in objects:
                    objects[key] = original(host, **kwargs)
                    counters[name]["constructed"] += 1
                else:
                    counters[name]["reused"] += 1
                # Members own the small container, sharing only static arrays.
                return copy.copy(objects[key])
        return classmethod(load)

    try:
        for name in specifications:
            cls = getattr(containers, name)
            changes.append((cls, cls.__dict__["from_host"]))
            cls.from_host = factory(name, cls.from_host)
        yield counters
    finally:
        for cls, original in changes:
            cls.from_host = original
        objects.clear()


@contextmanager
def serial_netcdf():
    """Hold one lock across each Dataset lifetime, including fallback writes."""
    import netCDF4
    import cftime
    lock = threading.RLock()
    changes = []
    replacements = {}

    class DatasetOwner:
        def __init__(self, factory, args, kwargs):
            lock.acquire()
            try:
                self.raw = factory(*args, **kwargs)
            except BaseException:
                lock.release()
                raise
            self.closed = False

        def __getattr__(self, name):
            return getattr(self.raw, name)

        def __enter__(self):
            return self.raw

        def close(self):
            if not self.closed:
                try:
                    self.raw.close()
                finally:
                    self.closed = True
                    lock.release()

        def __exit__(self, *exc):
            self.close()

    def wrap(original, dataset):
        def call(*args, **kwargs):
            if dataset:
                return DatasetOwner(original, args, kwargs)
            with lock:
                return original(*args, **kwargs)
        return call

    try:
        for module in (netCDF4, cftime):
            for name in ("Dataset", "MFDataset", "num2date", "date2num", "date2index", "num2pydate"):
                if hasattr(module, name):
                    original = getattr(module, name)
                    replacement = replacements.setdefault(id(original), wrap(original, name in ("Dataset", "MFDataset")))
                    changes.append((module, name, original))
                    setattr(module, name, replacement)
        for name, module in tuple(sys.modules.items()):
            if module is None or not name.startswith(("hexcore.", "woof.")):
                continue
            for attr, value in tuple(vars(module).items()):
                if id(value) in replacements:
                    changes.append((module, attr, value))
                    setattr(module, attr, replacements[id(value)])
        yield
    finally:
        reverse = {id(value): original for _module, _name, original in changes
                   for value in [replacements.get(id(original))] if value is not None}
        for name, module in tuple(sys.modules.items()):
            if module is not None and name.startswith(("hexcore.", "woof.")):
                for attr, value in tuple(vars(module).items()):
                    if id(value) in reverse:
                        setattr(module, attr, reverse[id(value)])
        for module, name, original in reversed(changes):
            setattr(module, name, original)


def roster_requests(arguments, door):
    """A roster supplies only member forcing and destinations; flags stay shared."""
    requests = []
    for run_id, member in roster_members(arguments, door):
        if getattr(member, "analysis", None) is not None:
            # The member's bundle is checked and rewritten into its own --init (and IAU inputs) by the same door
            # code a one-forecast --analysis run takes; its derived files go to <out>/<run_id>.analysis-work.
            from .analysis_door import AnalysisDoorRefusal, resolve_analysis

            try:
                member.analysis_receipt = resolve_analysis(member, registry_loader=lambda: door.load_registry(
                    door.resolve_repo(getattr(member, "repo", None))))
            except AnalysisDoorRefusal as error:
                raise MultiMemberRefusal(f"member {run_id}: {error}") from None
        requests.append((run_id, door.resolve_request(member)))
    first = requests[0][1]
    # Binding and the compiled kernel family are process-wide. Different
    # meshes belong to separate groups rather than rebinding a live driver.
    for _run_id, request in requests:
        if request.preflight or getattr(arguments, "samehost_devices", None) is not None:
            raise MultiMemberRefusal("resident members cannot combine preflight or partition-rank execution")
        if request.mesh != first.mesh or request.grid != first.grid or request.static != first.static:
            raise MultiMemberRefusal("resident members need the same pinned mesh, grid and static files")
    return requests


def roster_members(arguments, door):
    """Each roster record as the member's own forecast arguments: ``[(run_id, namespace)]``.

    Shared by the WOOF roster (``roster_requests``) and the MPAS suite roster (``run_suite_members``), so both
    routes read the same records the same way.
    """
    records = json.loads(Path(arguments.members).read_text(encoding="utf-8"))
    if not isinstance(records, list) or not 1 <= len(records) <= 4:
        raise MultiMemberRefusal("--members needs a JSON array of 1 to 4 member records")
    def stochastic_options(value):
        return {} if value is None else (json.loads(value) if isinstance(value, str) else dict(value))
    members, ids = [], set()
    # "analysis" names the member's own analysis bundle (woof.hex.analysis_bundle): the member then starts from it
    # exactly as a one-forecast --analysis run does (direct insertion, or its background with the increment over
    # the window).  A roster-wide --analysis starts every member that names no init or bundle of its own.
    allowed = {"run_id", "init", "lbc_dir", "init_source", "preset", "analysis",
               "start_time", "hours", "history_from_minutes", "checkpoint_hours",
               "restart_from", "stochastic_seed", "stochastic"}
    for record in records:
        if not isinstance(record, dict) or set(record) - allowed:
            raise MultiMemberRefusal(f"member record contains an unsupported key; allowed: {sorted(allowed)}")
        run_id = record.get("run_id", "")
        if (not isinstance(run_id, str) or not run_id or run_id in (".", "..")
                or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in run_id)
                or run_id in ids):
            raise MultiMemberRefusal("member run_id must be a unique safe path component")
        ids.add(run_id)
        member = copy.copy(arguments)
        member.members = None
        for name, value in record.items():
            if name != "run_id":
                setattr(member, name, Path(value) if name == "restart_from" else value)
        if 'stochastic' in record:
            # A member overrides selected controls, retaining the shared seed,
            # spectrum and other settings it did not name, even in a solo group.
            member.stochastic = {**stochastic_options(getattr(arguments, 'stochastic', None)),
                                 **stochastic_options(record['stochastic'])}
        from .stochastic import member_number, member_options
        explicit = stochastic_options(getattr(member, 'stochastic', None))
        # A singleton group still names the same member. Card grouping must
        # not replace its random stream with the scalar forecast's member 0.
        identity = explicit.pop('member', member_number(run_id))
        member.stochastic = member_options(identity, source_members=len(records),
                                           ensemble_size=len(records), **explicit)
        member.out = Path(arguments.out) / run_id
        member.scratch = Path(arguments.scratch) / run_id
        member.receipt = None
        members.append((run_id, member))
    return members


class _Prepared(BaseException):
    pass


def adopt_case_pins(source, before, target):
    """Copy into ``target`` every value the preparation rebound on ``source`` (the shared proof module).

    THE BREAKAGE THIS PREVENTS: the mesh binding wraps the driver's prepare_forecast_host in a closure
    whose globals are the original modules, so a member's preparation measured its init's case pins
    (surface classification, census, p_top, carrier pins) into the shared proof module and the member's
    private proof kept the frozen reference mesh's values; the first history capture of every member
    then stopped with "surface classification changed".
    """
    missing = object()
    for name, value in vars(source).items():
        if name.startswith("__") or isinstance(value, (FunctionType, ModuleType, type)):
            continue
        if before.get(name, missing) is not value:
            setattr(target, name, value)


class PreparedMembers:
    """Reuse an identical sealed host preparation within this one roster."""

    def __init__(self):
        self.entries = {}
        self.rows = []

    def prepare(self, member, original, paths, authority, *, _shared_proof=None, **kwargs):
        # The host preparation reads initialization and boundary metadata.
        # Bind every file's full bytes as well as every configuration input.
        inputs = {name: str(path) for name, path in paths.items()}
        for path in kwargs.get("lbc_paths") or ():
            with Path(path).open("rb") as stream:
                inputs[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
        key = hashlib.sha256(json.dumps([inputs, authority, kwargs],
                                       sort_keys=True, default=str).encode()).hexdigest()
        started = time.perf_counter()
        if key in self.entries:
            host, proof = self.entries[key]
            try:
                private_host = copy.deepcopy(host)
            except (TypeError, ValueError) as error:
                # A foreign sealed object without a safe clone is prepared
                # independently. Never alias mutable member state to save work.
                self.rows.append({"key": key, "mode": "independent",
                    "reason": f"cannot safely clone host state: {type(error).__name__}: {error}"})
            else:
                member.proof = private_module(proof)
                self.rows.append({"key": key, "mode": "shared preparation, private state",
                                  "seconds": time.perf_counter() - started})
                return private_host
        before = None if _shared_proof is None else dict(vars(_shared_proof))
        host = original(paths, authority, **kwargs)
        if before is not None:
            adopt_case_pins(_shared_proof, before, member.proof)
        self.entries[key] = (host, private_module(member.proof))
        self.rows.append({"key": key, "mode": "prepared",
                          "seconds": time.perf_counter() - started})
        return host


def run_members(arguments, door):
    from .device_admission import admit_co_tenancy
    arguments = copy.copy(arguments)
    if getattr(arguments, "out", None) is None:
        raise MultiMemberRefusal("--out is required for a member roster; each member needs a distinct history destination")
    if getattr(arguments, "scratch", None) is None:
        arguments.scratch = Path(str(arguments.out) + ".forecast-scratch")
    root, scratch = Path(arguments.out).expanduser().absolute(), Path(arguments.scratch).expanduser().absolute()
    if root.exists() or scratch.exists() or root.resolve() in scratch.resolve().parents or scratch.resolve() in root.resolve().parents:
        raise MultiMemberRefusal("member output and scratch must be absent and disjoint; prevents overwriting member artifacts")
    if str(scratch.resolve()) == str(root.resolve()):
        raise MultiMemberRefusal("member output and scratch must be different directories")
    for name, path in (("--out", root), ("--scratch", scratch)):
        if not path.parent.is_dir():
            raise MultiMemberRefusal(f"{name} {path} cannot be created: its parent {path.parent} does not exist.  Create "
                                     "the parent first; a mistyped deep path would otherwise look like a new roster")
    # The roster's own directories are the members' parents (<out>/<run_id>), so they exist before the members
    # are resolved.  THE BREAKAGE THIS PREVENTS: the members were resolved first, the door refused each member
    # because <out> did not exist, and with <out> made beforehand the roster refused because <out> existed, so
    # no roster could run at all.  A roster refused before execution removes the two directories it made.
    root.mkdir()
    scratch.mkdir()
    try:
        if suite_route(arguments, door):
            return run_suite_members(arguments, door, root, scratch)
        return _run_members(arguments, door, root, scratch, admit_co_tenancy)
    except _RosterRefused as refused:
        import shutil

        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(scratch, ignore_errors=True)
        raise refused.error from None


def suite_route(arguments, door) -> bool:
    """Whether the roster's shared flags name an MPAS suite route (``--physics-suite``, its namelist, or
    ``--physics-backend mpasphys``), decided by the same ``native_route`` a one-member forecast takes.

    THE BREAKAGE THIS PREVENTS: a roster (``global --members``, ``s2s``) went straight to the WOOF column
    seam (``resolve_request`` -> ``physics_suites.resolve_physics``), so ``--forecast-arg=--physics-suite=
    convection_permitting`` was refused in 0.5 s ("config_radt_cld_scheme='cld_fraction' is refused") while the
    same flags on one member ran the MPAS port.
    """
    from .mpasphys.forecast_v842 import native_route

    try:
        config, _text = native_route(arguments, default_mode=getattr(door, "PHYSICS_MODE_DEFAULT", "correct"))
    except ValueError as error:
        raise _RosterRefused(error) from error
    return config is not None


def argv_physics_route(argv) -> str:
    """``"mpas"`` or ``"arwen"``: the route a forecast-door argv (``woof hex forecast ...``) runs, decided the way the
    door decides it (its parser, the hex default physics, then :func:`suite_route`).

    For the planners that price a member before the door runs (``global_ensemble.group_limit``).  An argv the door
    would refuse is priced as the MPAS route, the larger one; the door then refuses it by name at launch."""
    import argparse

    from . import forecast_door, physics_default

    tokens = [str(token) for token in argv or ()]
    if tokens and tokens[0] == "forecast":
        tokens = tokens[1:]
    parser = argparse.ArgumentParser(add_help=False)
    forecast_door.add_forecast_arguments(parser)
    try:
        arguments, _unknown = parser.parse_known_args(tokens)
        physics_default.apply(arguments)
        return "mpas" if suite_route(arguments, forecast_door) else "arwen"
    except (Exception, SystemExit):
        return "mpas"


def _member_cells(member) -> int:
    from .mesh_rows import _dimensions

    for name in ("grid", "init"):
        path = getattr(member, name, None)
        if path:
            return int(_dimensions(Path(path))["nCells"])
    raise MultiMemberRefusal("a resident MPAS-suite group needs --grid (or a member init) to price its members")


def admit_suite_group(members, gate) -> dict | None:
    """Refuse an MPAS-suite group the card cannot hold, before any member starts (U-H4).

    Each member is charged the MPAS route's own floor (``device_admission.route_required_free_bytes``) plus its
    stochastic arrays, through the same ``admit_co_tenancy`` the WOOF roster decides on, so a group that does not
    fit is refused with "co-tenancy refuses" and the ensemble door splits it.  THE BREAKAGE IT PREVENTS: this
    roster started every member at once with no memory decision; at 30 km two MPAS members per RTX PRO 6000 ran
    out of memory ~3 min in.  A single member is not decided here: the one-member door admits it as it always has.
    The card is read in a child process (``global_ensemble._probe_cards``), so this process opens no CUDA context
    on the card its members need; behind a card gate the decision is the card's capacity (the group before may
    still hold it), as in the WOOF roster.  None when one member, or when no card can be read here."""
    if len(members) < 2:
        return None
    from . import forecast_door as door
    from .device_admission import admit_co_tenancy, card_profile_from_attributes, model_for_card
    from .global_ensemble import _probe_cards

    cards = _probe_cards(None)
    if not cards:
        return None
    card = cards[0]
    first = members[0][1]
    cells = _member_cells(first)
    levels = 55
    init = getattr(first, "init", None)
    if init:
        interfaces = door._vertical_interfaces(Path(init))
        levels = 55 if interfaces is None else int(interfaces) - 1
    profile = card_profile_from_attributes(card["name"], {"MultiProcessorCount": card["sms"],
                                                          "MaxThreadsPerMultiProcessor": card["threads"]})
    model = model_for_card(profile, "global", levels, rollback_armed=bool(getattr(first, "stop_on_refusal", False)))
    headroom = getattr(first, "headroom_mib", None)
    margin = None if headroom is None else int(round(float(headroom) * door.MIB))
    from .device_admission import route_required_free_bytes

    solo = route_required_free_bytes(cells, model, margin, route="mpas")
    additional = tuple(door.stochastic_array_bytes(cells, getattr(member, "stochastic", None), levels=levels)
                       for _, member in members)
    total = int(card["total"])
    return admit_co_tenancy(mesh_kind="point-cull" if getattr(first, "lbc_dir", None) else "global",
                            cells=cells, levels=levels, members=len(members), card_bytes=total,
                            free_bytes=total if gate is not None else int(card["free"]),
                            solo_required_bytes=solo, member_additional_bytes=additional)


def _suite_member_main(member, log_path) -> int:
    """One MPAS-suite member, in its own process: exactly the one-member ``woof hex forecast`` door."""
    import sys as _sys

    log = open(log_path, "a", buffering=1, encoding="utf-8")
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    _sys.stdout = _sys.stderr = log
    from . import cli
    from .errors import MpasPortError

    try:
        cli._select_numerics(member)
        return int(member.handler(member))
    except (MpasPortError, OSError, ValueError) as error:
        print(f"woof hex: {error}", file=_sys.stderr, flush=True)
        return 2


def _suite_member_process(member, log_path, member_memory="single"):
    # Set in the child before forecast construction, without mutating the parent.
    os.environ["WOOF_HEX_MEMBER_MEMORY"] = member_memory
    raise SystemExit(_suite_member_main(member, log_path))


def run_suite_members(arguments, door, root, scratch):
    """A roster on the MPAS suite route: every member runs the one-member MPAS route, all at once on this card.

    Each member is its own process (the MPAS route keeps process-wide framework state, so members cannot share
    one interpreter the way the WOOF roster's private modules do) started together, so the group is co-resident
    on the card exactly as the WOOF roster's group is.  The card gate (``woof.hex.card_handoff``) is held by this
    process for the group, so the ensemble door's pipelining still sees one group per card.  A member writes the
    same history, receipts and products a one-member ``forecast --physics-suite`` run writes, in
    ``<out>/<run_id>``; its log is ``<out>/<run_id>.log``.
    """
    import multiprocessing

    from .card_handoff import CardGate

    gate = CardGate.from_environ()
    try:
        members = roster_members(arguments, door)
        admission = admit_suite_group(members, gate)
    except Exception as error:
        raise _RosterRefused(error) from error
    # The members inherit no gate: this process holds it for the whole group.
    from .card_handoff import GATE_ENV, TICKET_ENV
    environment = {key: os.environ.pop(key) for key in (GATE_ENV, TICKET_ENV) if key in os.environ}
    context = multiprocessing.get_context("spawn")
    started = time.time()
    result = {"schema": "gpuwm-hex.multi-member/v1", "route": "mpas-suite", "status": "incomplete",
              "admission": admission, "members": [], "default_enabled": False}
    failed = 0
    try:
        if gate is not None:
            gate.acquire()
        processes = []
        for run_id, member in members:
            log = root / f"{run_id}.log"
            process = context.Process(target=_suite_member_process, args=(member, str(log),
                                              "shared-card" if len(members) > 1 else "single"),
                                      name=f"hex-suite-member-{run_id}")
            process.start()
            processes.append((run_id, member, log, process))
        for run_id, member, log, process in processes:
            process.join()
            rc = int(process.exitcode if process.exitcode is not None else 1)
            failed += rc != 0
            result["members"].append({"run_id": run_id, "returncode": rc, "out": str(member.out), "log": str(log)})
            print(f"woof hex: member {run_id} (MPAS suite route) exited {rc}; log {log}", file=sys.stderr, flush=True)
            if rc != 0 and log.is_file():
                tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
                print("\n".join(f"  {run_id}| {line}" for line in tail), file=sys.stderr, flush=True)
    finally:
        os.environ.update(environment)
        if gate is not None:
            gate.release()
            result["card_handoff"] = gate.receipt
    result.update(status="complete" if not failed else "failed", wall_seconds=round(time.time() - started, 3))
    (root / "multi-member-receipt.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if not failed else 1


class _RosterRefused(Exception):
    """A roster stopped before any member executed (its directories hold nothing worth keeping)."""

    def __init__(self, error):
        super().__init__(str(error))
        self.error = error


def publish_forecast_receipt(door, request, observed, binding, architecture, seconds, status):
    """Publish the same discovery contract as a solo forecast beside this member's frames."""
    receipt = door.build_receipt(
        request=request, admission=None, bind_receipt=binding, driver_receipt=observed,
        history=door.history_files(request.out), driver_argv=door.build_driver_argv(request),
        seconds=seconds, status=status, architecture=architecture)
    door._write_receipt(request, receipt)


def _run_members(arguments, door, root, scratch, admit_co_tenancy):
    from .card_handoff import CardGate
    gate = CardGate.from_environ()
    try:
        requests = roster_requests(arguments, door)
        import cupy as cp
        from . import cuda_backend
        first = door._with_resolved_model(requests[0][1])
        levels = int(first.model.levels)
        free, total = map(int, cp.cuda.runtime.memGetInfo())
        mesh_kind = "point-cull" if first.lbc_dir is not None else "global"
        # Behind a card gate the group before this one may still hold the card, so this first decision is
        # the card's capacity (does the group fit an empty card; a group that does not is split); the
        # free-memory decision is taken at the hand-off below, once the gate is held.
        admission = admit_co_tenancy(mesh_kind=mesh_kind, cells=first.cells,
            levels=levels, members=len(requests), card_bytes=total, free_bytes=total if gate is not None else free,
            solo_required_bytes=door.required_free_bytes(first.cells, model=first.model,
                                                          margin_bytes=first.headroom_bytes),
            member_additional_bytes=tuple(door.stochastic_array_bytes(request.cells, request.stochastic, levels=levels)
                                          for _, request in requests))
        architecture = door.admit_architecture()
        binding, driver = door._load_drivers(first.repo)
        bind_receipt = door._bind(binding, driver, first)
    except Exception as error:
        raise _RosterRefused(error) from error
    started = time.time()
    tables = SharedTables(scratch / "shared-tables", policy=arguments.member_buffers,
                          creation_manifest=scratch / "shared-buffer-created-paths.txt")
    prepared = []
    preparation = PreparedMembers()
    result = {"schema": "gpuwm-hex.multi-member/v1", "status": "incomplete",
              "admission": admission, "mesh_binding": bind_receipt,
              "default_enabled": False, "members": []}
    physical_card = os.environ.get("GPU_MUTEX_CARDS", "").split(",")[0]
    if not physical_card.isdecimal():
        physical_card = str(door.time_ledger.card_index())
    card = int(physical_card) if physical_card.isdecimal() else None
    # Keep each stream's reusable blocks until the resident group drains.
    # Physics phase release acts on the process default pool; sharing that
    # pool made one member evict blocks that another member immediately needs.
    resident_pool = cp.cuda.MemoryPool()
    family = kernel_cache_family(cuda_backend.KernelCache)
    old_cache = cuda_backend.KernelCache
    construction_lock = threading.Lock()
    ready = threading.Barrier(len(requests))
    try:
        # All preparation and source/pin checks use the real driver's main.
        # Intercept only its handoff to execution, preserving the loop body.
        for run_id, request in requests:
            member = private_module(driver)
            member.proof = private_module(driver.proof)
            from .pbl_cadence import pbl_cadence_decision
            member.PBL_CADENCE_DECISION = pbl_cadence_decision(
                dt_seconds=float(member.DT_SECONDS), requested=request.pbl_cadence)
            prepare_original = member.prepare_forecast_host
            def prepare(paths, authority, _member=member, _original=prepare_original, **kwargs):
                return preparation.prepare(_member, _original, paths, authority, _shared_proof=driver.proof,
                                           **kwargs)
            member.prepare_forecast_host = prepare
            payload = {}
            def retain(**kwargs):
                payload.update(kwargs)
                raise _Prepared()
            member.execute_forecast = retain
            with door.time_ledger.member(run_id, card=card, root=request.out):
                door.time_ledger.emit("forecast.construct", "begin")
                try:
                    member.main(door.build_driver_argv(request))
                except _Prepared:
                    pass
                door.time_ledger.emit("prep.member_ready")
            if not payload:
                raise MultiMemberRefusal("member driver did not prepare an executable forecast")
            if int(member.N_LEVELS) != levels:
                raise MultiMemberRefusal("member vertical level count disagrees with its admitted column extent")
            if len(requests) > 1:
                # One member has nothing to share: interning would hash and copy every table on one core
                # (3.6 s on the 60 km globe, card idle) for no reader.
                payload["host"] = tables.host(payload["host"])
            # The member's tail (history drain, final identity hash) runs after the card is released.
            member.DEFER_TAIL = True
            # The full group floor was checked before any stack allocation.
            # Each unchanged loop records that same collective decision.
            member.proof.gpu_memory_admission = lambda *args, **kwargs: dict(admission)
            original_construct = member.proof._construct_device_stack
            def construct(_original=original_construct, **kwargs):
                with construction_lock:
                    stack = _original(**kwargs)
                    cp.cuda.get_current_stream().synchronize()
                ready.wait(timeout=600)
                return stack
            member.proof._construct_device_stack = construct
            prepared.append((run_id, request, member, payload))
        # Preparation may allocate a CUDA context, so repeat collective
        # admission at the actual handoff to resident construction.
        def readmit():
            free, total = map(int, cp.cuda.runtime.memGetInfo())
            return admit_co_tenancy(mesh_kind=mesh_kind, cells=first.cells,
                levels=levels, members=len(requests), card_bytes=total, free_bytes=free,
                solo_required_bytes=admission["solo_required_bytes"],
                member_additional_bytes=tuple(admission["member_additional_bytes"]))
        if gate is None:
            admission.update(readmit())
        else:
            from .card_handoff import readmit_seconds
            wait_mono, wait_unix = time.monotonic(), time.time()
            gate.acquire()
            # The previous group released the gate once its memory was returned; its process may still be
            # exiting, so the free-memory decision is retaken until the card has the room (or refuses by
            # the same sentence after the timeout).
            deadline = time.monotonic() + readmit_seconds()
            while True:
                try:
                    admission.update(readmit())
                    break
                except MemoryError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(1.0)
            gate.receipt["admitted_after_seconds"] = round(time.monotonic() - wait_mono, 3)
            for run_id, request, _member, _payload in prepared:
                with door.time_ledger.member(run_id, card=card, root=request.out):
                    door.time_ledger.emit("prep.card_wait", "begin", t_mono=wait_mono, t_unix=wait_unix)
                    door.time_ledger.emit("prep.card_wait", "end", attrs=dict(gate.receipt))
            result["card_handoff"] = gate.receipt
        result["shared_buffers"] = tables.receipt()
        result["host_preparation"] = list(preparation.rows)
        cuda_backend.KernelCache = family
        from .drivers.samehost_physics_scope_v841 import scoped_physics_resources
        with serial_netcdf():
            with scoped_physics_resources() as physics, shared_device_tables() as device_tables, \
                    capture_exclusion(family.family_lock):
                def execute(index, item):
                    run_id, request, member, payload = item
                    stream = cp.cuda.Stream(non_blocking=True)
                    try:
                        with cp.cuda.Device(0), stream, cp.cuda.using_allocator(resident_pool.malloc), \
                                physics.rank(index, 0), door.time_ledger.member(run_id, card=card, root=request.out):
                            door.time_ledger.emit("prep.shared_buffers", attrs={
                                "buffers": len(tables.rows),
                                "storage": sorted({row["storage"] for row in tables.rows}),
                                "fallback": sorted({row["fallback"] for row in tables.rows if row["fallback"]})})
                            forecast = driver.execute_forecast
                            # Rebind the unchanged execute function to the private
                            # member globals, including its private proof module.
                            execute_function = FunctionType(forecast.__code__, vars(member), forecast.__name__, forecast.__defaults__, forecast.__closure__)
                            execute_function.__kwdefaults__ = forecast.__kwdefaults__
                            member_started = time.monotonic()
                            observed = execute_function(**payload)
                            return observed, member_started, door.time_ledger.open_stages()
                    except BaseException:
                        _member_failure(request)
                        ready.abort()
                        raise
                    finally:
                        # Drain this member before releasing shared objects or
                        # restoring physics factories, including failure paths.
                        with cp.cuda.Device(0):
                            stream.synchronize()
                with ThreadPoolExecutor(max_workers=len(prepared), thread_name_prefix="hex-member") as pool:
                    futures = [pool.submit(execute, index, item) for index, item in enumerate(prepared)]
                    executed = [future.result() for future in futures]
            # Every member's device work is done and its device stack is gone with execute_forecast's frame.
            # Return the memory and hand the card to the next group now, BEFORE the host-only tail (history
            # drain, final identity hash, authority checks, receipts), so the card never waits on it.
            result["resident_allocator"] = _allocator_receipt(resident_pool)
            resident_pool.free_all_blocks()
            result["resident_allocator"]["reserved_bytes_after_release"] = int(resident_pool.total_bytes())
            if gate is not None:
                from .card_handoff import release_device_memory
                gate.release(**release_device_memory(cp))

            def finish(item, observed, member_started, open_stages):
                run_id, request, member, payload = item
                try:
                    with door.time_ledger.member(run_id, card=card, root=request.out, open_stages=open_stages):
                        tail = observed.pop("_deferred_tail", None)
                        if tail is not None:
                            tail()
                        member.proof.measure_execution_sources()
                        if member.verify_forecast_authorities({"grid": request.grid, "static": request.static, "init": request.init}) != payload["authority_receipt"]:
                            raise RuntimeError("member authority bytes changed during execution")
                        from .physics_backend_admission import resolve_backend
                        backend = resolve_backend(payload["host"]["physics_backend"])
                        member.proof.verify_arwen_checkout_git(payload["arwen_checkout"], manifest=member._manifest_for_run(backend, payload["arwen_checkout"]))
                        status = "passed" if not observed.get("refusal") else "truncated_by_model_refusal"
                        row = {"run_id": run_id, "card": card, "status": status, "forecast": observed}
                        (request.out / "multi-member-receipt.json").write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
                        publish_forecast_receipt(door, request, observed, bind_receipt, architecture,
                                                 time.monotonic() - member_started, status)
                        door.time_ledger.end_if_open("post.forecast")
                        door.time_ledger.emit("finish", attrs={"status": status})
                        return row
                except BaseException:
                    _member_failure(request)
                    raise
            with ThreadPoolExecutor(max_workers=len(prepared), thread_name_prefix="hex-member-tail") as pool:
                futures = [pool.submit(finish, item, *done) for item, done in zip(prepared, executed)]
                result["members"] = [future.result() for future in futures]
        result["physics_resource_scope"] = physics.receipt()
        result["device_tables"] = copy.deepcopy(device_tables)
        result["status"] = "passed" if all(row["status"] == "passed" for row in result["members"]) else "failed"
    finally:
        cuda_backend.KernelCache = old_cache
        if gate is not None:
            gate.release()  # a failed group hands the card on too (the kernel would at exit anyway)
        try:
            if "resident_allocator" not in result:
                result["resident_allocator"] = _allocator_receipt(resident_pool)
            # Executor shutdown and each worker's finally have drained all streams,
            # including on failure, before any cached member blocks are returned.
            resident_pool.free_all_blocks()
            result["resident_allocator"]["reserved_bytes_after_release"] = int(resident_pool.total_bytes())
        finally:
            tables.close()
            result["shared_buffers"] = tables.receipt()
            result["wall_seconds"] = time.time() - started
            (root / "multi-member.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return 0 if result["status"] == "passed" else 1


def _allocator_receipt(pool):
    return {"kind": "scoped-caching-pool",
            "binding": "thread-local allocator, private member streams",
            "used_bytes_at_drain": int(pool.used_bytes()),
            "reserved_bytes_at_drain": int(pool.total_bytes())}


def _member_failure(request):
    import traceback
    failure = request.out / "member-traceback.log"
    failure.parent.mkdir(parents=True, exist_ok=True)
    failure.write_text(traceback.format_exc(), encoding="utf-8")
