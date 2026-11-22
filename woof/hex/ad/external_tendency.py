# SPDX-License-Identifier: Apache-2.0
"""The external-tendency slot: a caller's tendency added to the physics rates once a model step.

What it is for: a learned correction (a network, or a plain table of numbers) adds a small tendency to the model at
every step, and the adjoint carries a forecast error back to that tendency at every step. The engine does not import
any learning library; a caller hands arrays over on the card (DLPack or ``__cuda_array_interface__``, no copies).

THE SLOT. Four physical tendencies at cell centres, float32, C-contiguous ``(nz, ncol)`` (row k = model level k
from the ground, column i = the engine's local solve cell i):

    theta  K s-1          joins rthblten
    qv     kg kg-1 s-1    joins rqvblten
    u      m s-2 zonal    joins rublten
    v      m s-2 merid.   joins rvblten

They are added inside ``physics_get_tend`` (woof/hex/mpasphys/rrfs/driver_rrfs.py ``ResidentDriverTendencies.run``;
the assembler route in resident_v842 / resident_rrfs the same way) after every scheme has written its rates and before
the edge projection (``project_edges_v841_f32``) and the coupling (``mpas_rk_cells_v842_f32``,
``mpas_rk_edges_v842_f32``) hand them to the dynamics. The adds land on scratch copies of the four rates; a scheme's
own rate array is never written. Because the add sits upstream of the coupling, the state-dependent parts (the mass
weighting, the projection to edges) are done by kernels that already carry checked derivative rules, and the add
itself is a pure identity: kind ``external_tendency_f32(int n, const float *ext, float *tend)``, window rule
``R::Axpy { c: 1 }`` (tangent t_tend += t_ext; reverse ext_bar += tend_bar).

ABSENT = NOTHING. With no callback installed and ``WOOF_HEX_EXTERNAL_TENDENCY`` unset, ``slot_for`` returns None and
nothing is allocated, copied or launched: every run, plan and test is the same bytes.

Two sources:
- ``install(callback)``: in process. ``callback(k, state) -> {field: array or None}`` once a step (k = 0-based index
  of the model step [t_k, t_k + dt] of this run; ``state`` the engine's fields at the step start, see ``StateView``).
  The engine synchronises its stream before the call and the device after it. ``save=path`` keeps every step's
  arrays on the host and writes them as a table (``write_table``) so a recording replays exactly that forecast.
- ``WOOF_HEX_EXTERNAL_TENDENCY=table.npz``: per field a float32 ``(nsteps, nz, ncol)`` array. Each field's table is
  ONE device allocation uploaded once, at the first step; step k's launch points at slice k. This is the form a
  recording uses: a callback is never run inside a traced process (its own kernels have no derivative row and would
  refuse the seal). After a window's reverse, the table allocation's cotangent IS dJ/d ext, every step at once, in
  the same layout.

``WOOF_HEX_EXTERNAL_TENDENCY_RECEIPT=path`` writes ``external-tendency.json`` (fields, shapes, the tables' and the
state fields' device pointers and bytes) when the slot is built, for the alloc map.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

KIND = "external_tendency_f32"
MODULE_KEY = "hexcore.ad.external_tendency"
ENV_TABLE = "WOOF_HEX_EXTERNAL_TENDENCY"
ENV_RECEIPT = "WOOF_HEX_EXTERNAL_TENDENCY_RECEIPT"
FIELDS = ("theta", "qv", "u", "v")
RATE_OF = {"theta": "rthblten", "qv": "rqvblten", "u": "rublten", "v": "rvblten"}

#: A zero word of ext leaves tend untouched, so a zero tendency keeps a -0 tendency word: bit for bit the run without
#: the slot (woof.hex.iau_forcing's hex_iau_axpy does the same). The derivative is the identity add either way.
SOURCE = r'''
extern "C" __global__ void external_tendency_f32(const int n, const float *ext, float *tend)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    const float x = ext[i];
    if (x != 0.0f) tend[i] = tend[i] + x;
}
'''
THREADS = 128

#: (name in the mapping, pool, pool field). Read at the step start (physics_get_tend: the schemes read these words).
STATE_FIELDS = (
    ("theta_m", "state", "theta_m"), ("rho_zz", "state", "rho_zz"),
    ("theta", "diag", "theta"), ("pressure", "diag", "pressure"),
    ("u_zonal", "diag", "uReconstructZonal"), ("u_meridional", "diag", "uReconstructMeridional"),
    ("psfc", "diag", "surface_pressure"), ("skintemp", "sfc_input", "skintemp"),
    ("t2m", "diag_physics", "t2m"), ("q2", "diag_physics", "q2"),
    ("hfx", "diag_physics", "hfx"), ("lh", "diag_physics", "lh"),
    # where each column is (radians) and, on a limited-area mesh, its boundary ring (0 = interior)
    ("lat", "mesh", "latCell"), ("lon", "mesh", "lonCell"), ("bdy_mask", "mesh", "bdyMaskCell"),
)

Callback = Callable[[int, "StateView"], Mapping[str, Any]]


@dataclass
class _Installed:
    callback: Callback
    fields: tuple[str, ...]
    save: str | None
    saved: dict[str, list] = field(default_factory=dict)
    calls: int = 0


_INSTALLED: _Installed | None = None


def install(callback: Callback, *, fields: tuple[str, ...] = FIELDS, save: str | None = None) -> None:
    """Install CALLBACK for the forecasts this process builds from now on (see the module text)."""
    global _INSTALLED
    bad = [f for f in fields if f not in FIELDS]
    if bad:
        raise ValueError(f"external tendency fields {bad} are not slot fields {FIELDS}")
    if not callable(callback):
        raise TypeError("the external tendency callback must be callable: callback(k, state) -> {field: array}")
    _INSTALLED = _Installed(callback, tuple(fields), save)


def uninstall() -> None:
    global _INSTALLED
    _INSTALLED = None


def installed() -> _Installed | None:
    return _INSTALLED


def write_table(path: str | os.PathLike, saved: Mapping[str, list] | None = None) -> Path:
    """Write the arrays a ``save`` callback returned as a table (npz, per field float32 (nsteps, nz, ncol))."""
    saved = (_INSTALLED.saved if _INSTALLED is not None else {}) if saved is None else saved
    arrays = {name: np.stack(rows).astype(np.float32, copy=False) for name, rows in saved.items() if rows}
    if not arrays:
        raise ValueError("no external tendency was saved (no step ran, or every field was absent)")
    steps = {a.shape[0] for a in arrays.values()}
    if len(steps) != 1:
        raise ValueError(f"saved fields cover different step counts {sorted(steps)}")
    path = Path(path)
    np.savez(path, **arrays)
    return path


def _active_source() -> str | None:
    if _INSTALLED is not None:
        return "callback"
    return "table" if os.environ.get(ENV_TABLE) else None


class StateView(dict):
    """The engine's fields at the start of step k, as zero-copy CuPy views (read-only by convention), by name
    (STATE_FIELDS, plus ``qv``). Attributes: ncol, nz, k, dt, time_seconds, stream_ptr."""

    def __init__(self, arrays: Mapping[str, Any], **attrs: Any):
        super().__init__(arrays)
        self.__dict__.update(attrs)


def _as_device(cp: Any, value: Any, name: str, shape: tuple[int, int]) -> Any:
    if value is None:
        return None
    if isinstance(value, cp.ndarray):
        array = value
    elif hasattr(value, "__cuda_array_interface__"):
        array = cp.asarray(value)  # zero-copy view of the producer's memory
    elif hasattr(value, "__dlpack__"):
        array = cp.from_dlpack(value)
    else:
        raise TypeError(f"external tendency {name!r}: {type(value).__name__} is not a device array "
                        "(give a CuPy array or a CUDA torch tensor)")
    if array.dtype != np.float32:
        raise TypeError(f"external tendency {name!r} is {array.dtype}, not float32")
    if tuple(array.shape) != shape:
        raise ValueError(f"external tendency {name!r} has shape {tuple(array.shape)}, not (nz, ncol) = {shape}")
    if not array.flags.c_contiguous:
        raise ValueError(f"external tendency {name!r} is not C-contiguous")
    if array.device.id != cp.cuda.Device().id:
        raise ValueError(f"external tendency {name!r} is on card {array.device.id}, the engine on "
                         f"{cp.cuda.Device().id}")
    return array


class Slot:
    """One assembler's slot: the targets (scratch rate copies) and, per step, the adds."""

    def __init__(self, *, cp: Any, cache: Any, shape: tuple[int, int], provider: Any = None, dt: float | None = None,
                 qv_index: int | None = None):
        self.cp, self.cache, self.shape = cp, cache, (int(shape[0]), int(shape[1]))
        self.provider, self.dt, self.qv_index = provider, dt, qv_index
        self.time_seconds: float | None = None
        self.source = _active_source()
        self.k = 0
        self.launches = 0
        self.tables: dict[str, Any] = {}
        self.nsteps = None
        self.state_names: list[str] = []
        self._state: dict[str, Any] | None = None
        self.kernel = cache.raw_kernel(KIND, SOURCE, module_key=MODULE_KEY)
        if self.source == "table":
            self._load_table(os.environ[ENV_TABLE])
            self.fields = tuple(f for f in FIELDS if f in self.tables)
        else:
            self.fields = _INSTALLED.fields
        self._receipt()

    def _load_table(self, path: str) -> None:
        loaded = _PRELOADED.get(path)
        if loaded is not None:
            for name, table in loaded.items():
                if table.ndim != 3 or tuple(table.shape[1:]) != self.shape:
                    raise ValueError(f"{ENV_TABLE}={path}: {name} has shape {tuple(table.shape)}, not (nsteps, nz, "
                                     f"ncol) with (nz, ncol) = {self.shape}")
            steps = {t.shape[0] for t in loaded.values()}
            if len(steps) != 1:
                raise ValueError(f"{ENV_TABLE}={path}: fields cover different step counts")
            self.tables, self.nsteps = dict(loaded), steps.pop()
            return
        with np.load(path) as data:
            unknown = sorted(set(data.files) - set(FIELDS))
            if unknown:
                raise ValueError(f"{ENV_TABLE}={path}: arrays {unknown} are not slot fields {FIELDS}")
            for name in FIELDS:
                if name not in data.files:
                    continue
                host = np.ascontiguousarray(data[name], dtype=np.float32)
                if host.ndim != 3 or tuple(host.shape[1:]) != self.shape:
                    raise ValueError(f"{ENV_TABLE}={path}: {name} has shape {host.shape}, not (nsteps, nz, ncol) "
                                     f"with (nz, ncol) = {self.shape}")
                if self.nsteps is None:
                    self.nsteps = host.shape[0]
                elif host.shape[0] != self.nsteps:
                    raise ValueError(f"{ENV_TABLE}={path}: fields cover different step counts")
                # One allocation per field, uploaded once; step k's launch points at slice k.
                self.tables[name] = self.cp.asarray(host)
        if not self.tables:
            raise ValueError(f"{ENV_TABLE}={path} holds none of the slot fields {FIELDS}")

    def state(self) -> dict[str, Any]:
        """The state fields (bound once: persistent pool arrays)."""
        if self._state is None:
            arrays: dict[str, Any] = {}
            p = self.provider
            if p is not None:
                old = getattr(p, "level", None)
                try:
                    p.level = 1
                    for name, pool, pool_name in STATE_FIELDS:
                        try:
                            arrays[name] = p.field(pool, pool_name)
                        except (KeyError, ValueError):
                            continue
                    try:
                        scalars = p.field("state", "scalars")
                        if self.qv_index is not None:
                            arrays["qv"] = scalars[self.qv_index]
                    except (KeyError, ValueError):
                        pass
                finally:
                    if old is not None:
                        p.level = old
            self._state = arrays
            self.state_names = sorted(arrays)
        return self._state

    def _receipt(self) -> None:
        out = {"source": self.source, "kind": KIND, "fields": list(self.fields), "shape": list(self.shape),
               "nsteps": self.nsteps,
               "tables": {n: {"ptr": int(a.data.ptr), "bytes": int(a.nbytes), "shape": list(a.shape)}
                          for n, a in self.tables.items()}}
        if self.source == "callback":
            out["state"] = {n: {"ptr": int(a.data.ptr), "bytes": int(a.nbytes), "shape": list(a.shape)}
                            for n, a in self.state().items()}
        out["pid"] = os.getpid()
        self.receipt = out
        path = os.environ.get(ENV_RECEIPT)
        if path:
            # One line per forecast process: a recording runs the forecast several times (warm, A, B), and the plan is
            # sealed from A, so every process's pointers are kept (allocation_ids maps them to the plan's ids).
            with open(path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(out) + "\n")

    def ext_for_step(self) -> dict[str, Any]:
        """{field: device array (nz, ncol)} for step self.k; absent fields are left out."""
        cp, k = self.cp, self.k
        if self.source == "table":
            if k >= self.nsteps:
                raise RuntimeError(f"{ENV_TABLE}: the table covers {self.nsteps} steps and step {k} was asked for; "
                                   "the run is longer than its external tendency")
            return {n: t[k] for n, t in self.tables.items()}
        hook = _INSTALLED
        stream = cp.cuda.get_current_stream()
        stream.synchronize()
        view = StateView(self.state(), ncol=self.shape[1], nz=self.shape[0], k=k, dt=self.dt,
                         time_seconds=self.time_seconds,
                         stream_ptr=int(stream.ptr))
        returned = hook.callback(k, view) or {}
        cp.cuda.Device().synchronize()
        unknown = sorted(set(returned) - set(hook.fields))
        if unknown:
            raise ValueError(f"the external tendency callback returned {unknown}, outside its fields {hook.fields}")
        out = {}
        for name in hook.fields:
            array = _as_device(cp, returned.get(name), name, self.shape)
            if array is None:
                continue
            out[name] = array
            if hook.save:
                hook.saved.setdefault(name, []).append(cp.asnumpy(array))
        hook.calls += 1
        return out

    def add(self, targets: Mapping[str, Any], *, dt: float | None = None, time_seconds: float | None = None) -> None:
        """Add this step's external tendency into TARGETS ({field: scratch rate (nz, ncol)}) and advance k. DT and
        TIME_SECONDS are the calling step's (``SlotCall.dt``, ``.time_seconds``): the callback's state view carries
        them, so a callback has the run's clock with no word of its own."""
        if dt is not None:
            self.dt = float(dt)
        if time_seconds is not None:
            self.time_seconds = float(time_seconds)
        n = self.shape[0] * self.shape[1]
        for name, ext in self.ext_for_step().items():
            target = targets[name]
            if tuple(target.shape) != self.shape or not target.flags.c_contiguous:
                raise ValueError(f"external tendency target {name} is {target.shape}, not a contiguous {self.shape}")
            self.kernel(((n + THREADS - 1) // THREADS,), (THREADS,), (np.int32(n), ext, target))
            self.launches += 1
        self.k += 1


_PRELOADED: dict[str, dict[str, Any]] = {}


def preload(cp: Any) -> None:
    """Upload the table named by WOOF_HEX_EXTERNAL_TENDENCY now, before the step loop (the forecast driver calls
    this after it has its card). Then each table is an allocation of boundary 0 of a recording, not an upload inside
    step 1, so a window can replace its words from boundary 0 (``--x0-dir`` dense blocks) and one recording made with
    a zero table serves every training iteration (19 Nov). Unset: nothing happens."""
    path = os.environ.get(ENV_TABLE)
    if not path or path in _PRELOADED:
        return
    with np.load(path) as data:
        unknown = sorted(set(data.files) - set(FIELDS))
        if unknown:
            raise ValueError(f"{ENV_TABLE}={path}: arrays {unknown} are not slot fields {FIELDS}")
        tables = {name: cp.asarray(np.ascontiguousarray(data[name], dtype=np.float32))
                  for name in FIELDS if name in data.files}
    if not tables:
        raise ValueError(f"{ENV_TABLE}={path} holds none of the slot fields {FIELDS}")
    _PRELOADED[path] = tables


def allocation_ids(plan_dir: str | os.PathLike, receipt: str | os.PathLike) -> dict[str, Any]:
    """The plan allocation (id, byte offset) of each table and state field a recording's receipt names.

    PLAN_DIR is a sealed plan package (its ``catalog-a.json`` lists every allocation of recording A with its recorded
    base address and bytes); RECEIPT the ``WOOF_HEX_EXTERNAL_TENDENCY_RECEIPT`` file written during that recording
    (one line per forecast process). The line whose table pointers all fall inside A's allocations is A's. Returns
    {"pid", "tables": {field: {"allocation", "offset", "bytes"}}, "state": {...}}; a runner session reads slice k of
    a table's shadow at (allocation, offset + k nz ncol 4)."""
    catalog = json.loads((Path(plan_dir) / "catalog-a.json").read_text())["allocations"]
    spans = sorted(((int(a["recorded_base"]), int(a["bytes"]), int(a["id"]), a.get("freed_clock_ns"))
                    for a in catalog if a.get("memory", "device") == "device" and a.get("recorded_base") is not None),
                   key=lambda s: (s[0], s[3] is not None))

    def find(ptr: int, nbytes: int):
        hits = [s for s in spans if s[0] <= ptr and ptr + nbytes <= s[0] + s[1]]
        if not hits:
            return None
        base, size, ident, _ = min(hits, key=lambda s: (s[3] is not None, s[1]))   # live first, then the tightest
        return {"allocation": ident, "offset": ptr - base, "bytes": nbytes}

    for line in Path(receipt).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tables = {n: find(int(t["ptr"]), int(t["bytes"])) for n, t in (row.get("tables") or {}).items()}
        if tables and all(tables.values()):
            state = {n: find(int(t["ptr"]), int(t["bytes"])) for n, t in (row.get("state") or {}).items()}
            return {"pid": row.get("pid"), "tables": tables, "state": state}
    raise ValueError(f"no line of {receipt} has every table inside an allocation of {plan_dir}/catalog-a.json "
                     "(was the plan recorded with this table?)")


def slot_for(*, cp: Any, cache: Any, shape: tuple[int, int], provider: Any = None, dt: float | None = None,
             qv_index: int | None = None) -> Slot | None:
    """The slot for one physics_get_tend assembler, or None when no external tendency is given (the default)."""
    if _active_source() is None:
        return None
    return Slot(cp=cp, cache=cache, shape=shape, provider=provider, dt=dt, qv_index=qv_index)


_UNSET = object()


def rates_for(owner: Any, rates: dict[str, Any], *, cp: Any, cache: Any, mass: Any, provider: Any = None,
              qv_index: int | None = None, dt: float | None = None,
              time_seconds: float | None = None) -> dict[str, Any]:
    """The assembler route's entry: OWNER's slot (built at its first step and kept on it), then substitute_rates.
    No external tendency: RATES itself, and nothing is built."""
    slot = getattr(owner, "_external_tendency_slot", _UNSET)
    if slot is _UNSET:
        slot = slot_for(cp=cp, cache=cache, shape=tuple(mass.shape), provider=provider, qv_index=qv_index)
        owner._external_tendency_slot = slot
    return substitute_rates(slot, rates, cp, dt=dt, time_seconds=time_seconds)


def substitute_rates(slot: Slot | None, rates: dict[str, Any], cp: Any, *, dt: float | None = None,
                     time_seconds: float | None = None) -> dict[str, Any]:
    """The assembler route (resident_v842 / resident_rrfs get_tend without the resident helper): RATES with each
    slot field's rate replaced by a fresh copy that carries this step's external tendency. No slot: RATES itself."""
    if slot is None:
        return rates
    out = dict(rates)
    targets = {}
    for name in FIELDS:
        rate = rates.get(RATE_OF[name])
        if rate is None:
            raise ValueError(f"the external tendency slot joins {RATE_OF[name]}, which this physics route does not "
                             "carry (it needs the boundary-layer scheme on)")
        if tuple(rate.shape) != slot.shape:
            raise ValueError(f"{RATE_OF[name]} is {tuple(rate.shape)}, the slot {slot.shape}: this route's rate "
                             "arrays are not the solve-cell width the slot is defined on")
        out[RATE_OF[name]] = targets[name] = rate.copy()
    slot.add(targets, dt=dt, time_seconds=time_seconds)
    return out
