"""Reusable RK/acoustic launch graphs with caller-owned state staging.

No arithmetic is replaced. Boundary interpolation remains an eager launch of
the existing kernel so its binary32 clock arguments are refreshed on replay.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass, replace
import enum
import os
import time
import traceback
from types import SimpleNamespace
from typing import Any

import numpy as np

from .cuda_backend import runtime
from .cuda_graph_tables import GraphTables


_GRAPH_LAUNCH_SCOPE: ContextVar[bool] = ContextVar("hex_graph_launch_scope", default=False)


@contextmanager
def graph_launch_scope():
    """Keep launch order while omitting event timing during graph setup."""
    token = _GRAPH_LAUNCH_SCOPE.set(True)
    try:
        yield
    finally:
        _GRAPH_LAUNCH_SCOPE.reset(token)


@dataclass(frozen=True, slots=True)
class _UnmeasuredTiming(runtime.KernelTiming):
    measured: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {**super(_UnmeasuredTiming, self).as_dict(), "measured": False}


def launch_timed(kernel, grid, block, args, *, warmup=1, repeats=1, stream=None):
    """Use ordinary timing outside graph setup, and exact launches inside."""
    if not _GRAPH_LAUNCH_SCOPE.get():
        return runtime.launch_timed(kernel, grid, block, args,
                                    warmup=warmup, repeats=repeats, stream=stream)
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    cp = runtime._cupy()
    selected = cp.cuda.get_current_stream() if stream is None else stream
    for _ in range(1 + warmup + repeats):
        kernel(tuple(grid), tuple(block), tuple(args), stream=selected)
    return _UnmeasuredTiming(
        compile_seconds=runtime._KERNEL_COMPILE_SECONDS.get(id(kernel), 0.0),
        first_launch_ms=0.0, first_wall_ms=0.0, mean_launch_ms=0.0,
        min_launch_ms=0.0, max_launch_ms=0.0, repeats=repeats,
    )


def attach_subcycle_graphs(driver: Any) -> "SubcycleGraphs":
    """Attach the host controller to one already-admitted driver."""
    existing = getattr(driver, "subcycle_graphs", None)
    if existing is not None:
        return existing
    graphs = driver.subcycle_graphs = SubcycleGraphs(driver)
    eager = driver._advance_dynamics_subcycle_v841

    def advance(state, **kwargs):
        return graphs.advance(eager, state, **kwargs)

    driver._advance_dynamics_subcycle_v841 = advance
    return graphs


#: Meshes of this many cells and more run the dynamics subcycle eager by
#: default (the 2026-10-14 memory census, items 5 and 11).  A replay
#: reads and writes fixed addresses, so the graph holds a staged copy of every
#: subcycle input (12.0 KB per column at x1.40962) and keeps its private
#: scratch pool for the life of the run (19.4 KB per column), memory the
#: transport and physics between replays cannot use.  At x1.40962 on an RTX
#: 5090 replay bought nothing: 0.248 s per step against 0.245 eager (census
#: runs arwen120-clean2 and arwen120-nographs-clean2), and a bigger mesh is
#: even less launch-bound.  Since F1 (2026-10-15) replay is byte-identical to
#: eager (tools/graph_identity_gate.sh), so this choice moves memory and speed
#: only, never a bit.  Smaller meshes (regional cuts, the gate meshes) keep
#: replay.  ``WOOF_HEX_CUDA_GRAPHS=1`` forces replay on any mesh and ``=0``
#: forces eager.
GRAPH_EAGER_FROM_CELLS = 40_962


def graphs_enabled(cells: int | None = None) -> bool:
    """Whether the forecast attaches subcycle graph replay.

    ``WOOF_HEX_CUDA_GRAPHS``: ``1`` replays, ``0`` runs eager, unset or
    ``auto`` replays below :data:`GRAPH_EAGER_FROM_CELLS` cells (and always
    when the caller does not say how many).
    """
    from woof.hip_backend import supports_graphs
    try:
        capture = supports_graphs()
    except ModuleNotFoundError as error:
        # No CuPy means no HIP runtime either; the policy below is the CUDA one.  THE BREAKAGE THIS
        # PREVENTS (hex 0.4.12 final gate, public CI battery): the AMD fold made this policy query import
        # CuPy, so the CPU battery could no longer read the graph policy at all.
        if error.name != "cupy":
            raise
        capture = True
    if not capture:
        if os.environ.get("WOOF_HEX_CUDA_GRAPHS") == "1":
            raise RuntimeError("CuPy ROCm has no stream capture; use WOOF_HEX_CUDA_GRAPHS=0")
        return False
    value = os.environ.get("WOOF_HEX_CUDA_GRAPHS", "auto")
    if value != "auto":
        return value == "1"
    return cells is None or int(cells) < GRAPH_EAGER_FROM_CELLS


def graph_policy(cells: int) -> dict[str, Any]:
    """The receipt row for the replay choice on a mesh of ``cells``."""
    return {"setting": os.environ.get("WOOF_HEX_CUDA_GRAPHS", "auto"),
            "cells": int(cells), "eager_from_cells": GRAPH_EAGER_FROM_CELLS,
            "replay": graphs_enabled(cells)}


def _error_detail(error: BaseException) -> dict[str, Any]:
    return {
        "type": type(error).__name__, "message": str(error),
        "site": [f"{frame.filename}:{frame.lineno}:{frame.name}"
                 for frame in traceback.extract_tb(error.__traceback__)[-12:]],
        "cleanup_errors": list(getattr(error, "__notes__", ())),
    }


#: Host values a replay never needs at their capture-time value, so they stay
#: out of the graph key: ``time_seconds`` is rebound on the returned state,
#: ``h2d`` is upload bookkeeping, and ``dynamics_substep`` is read only by a
#: physics stage callback, which never replays (``advance``).
_UNKEYED_HOST_FIELDS = frozenset({"time_seconds", "h2d", "dynamics_substep"})

#: Host leaves keyed by value. A capture bakes every host value the eager body
#: reads into kernel arguments, so a value that changes after capture must
#: change the key (F1, 2026-10-15: keying by type alone let the lagged-heating
#: switch stay at its capture-time "off" for the whole run).
_HOST_VALUE_TYPES = (bool, int, float, complex, str, bytes, type(None),
                     np.generic, np.dtype, enum.Enum)


def _layout(value: Any, cp: Any, *, pointers: bool = False) -> Any:
    if isinstance(value, cp.ndarray):
        layout = (tuple(value.shape), str(value.dtype), tuple(value.strides))
        return (*layout, int(value.data.ptr)) if pointers else layout
    if is_dataclass(value):
        return (type(value).__qualname__, tuple(
            (f.name, _layout(getattr(value, f.name), cp, pointers=pointers))
            for f in fields(value)
            if f.name not in _UNKEYED_HOST_FIELDS
        ))
    if isinstance(value, dict):
        return tuple((k, _layout(v, cp, pointers=pointers)) for k, v in value.items()
                     if k not in _UNKEYED_HOST_FIELDS)
    if isinstance(value, SimpleNamespace):
        return (type(value).__qualname__, _layout(vars(value), cp, pointers=pointers))
    if isinstance(value, (tuple, list)):
        return tuple(_layout(v, cp, pointers=pointers) for v in value)
    if pointers or isinstance(value, _HOST_VALUE_TYPES):
        return (type(value).__qualname__, repr(value))
    return type(value).__qualname__


def _stage(value: Any, cp: Any, previous: Any = None) -> Any:
    if isinstance(value, cp.ndarray):
        target = cp.empty_like(value) if previous is None else previous
        cp.copyto(target, value)
        return target
    if is_dataclass(value):
        return replace(value, **{
            f.name: _stage(getattr(value, f.name), cp,
                           None if previous is None else getattr(previous, f.name))
            for f in fields(value)
        })
    if isinstance(value, dict):
        return {k: _stage(v, cp, None if previous is None else previous[k])
                for k, v in value.items()}
    if isinstance(value, SimpleNamespace):
        return type(value)(**_stage(vars(value), cp,
                           None if previous is None else vars(previous)))
    return value


class WithheldTendencyRead(RuntimeError):
    """The subcycle read a tendency-carrier field the graph does not stage."""


#: Carrier fields that are device arrays but are not staged: the held physics'
#: scalar block, which the v8.4.1 subcycle never reads (the scalar transport
#: after the subcycle reads the caller's own carrier). Staging it would copy
#: nScalars planes per subcycle, about 7 GB at 15 km. A driver whose subcycle
#: does read it (the RRFS driver's coupled source) runs eager, by name.
_WITHHELD_TENDENCY_ARRAYS = frozenset({"scalars"})


class _SubcycleTendencies(SimpleNamespace):
    """The physics-tendency carrier as the graph stages it.

    Every device array of the caller's carrier is staged and every field that
    is None stays None, so no field can fall out of the replayed subcycle. A
    field that appears after capture (F1: SG2's ``diabatic_theta``, None on
    step 1 and an array from step 2) changes the graph key, so the graph is
    captured again with it. The fields NOT staged -- the scalar block and every
    host value (clock, provenance, transfer counters), which a replay would
    freeze at capture -- raise on read instead of reading as absent:
    ``getattr(carrier, name, None)`` swallows only AttributeError, which is
    exactly how F1 stayed silent. The raise disables replay for that key and
    the subcycle runs eager, with the field named in the fallback receipt.
    """

    def __getattr__(self, name: str) -> Any:  # only names that are not staged
        if not name.startswith("__") and name in self.__dict__.get("_withheld", ()):
            raise WithheldTendencyRead(
                f"the dynamics subcycle read physics_tendencies.{name}, which "
                "CUDA graph replay does not stage"
            )
        raise AttributeError(name)


def _subcycle_tendencies(tendencies: Any, cp: Any) -> _SubcycleTendencies:
    if is_dataclass(tendencies):
        items = [(f.name, getattr(tendencies, f.name)) for f in fields(tendencies)]
    else:
        items = list(vars(tendencies).items())
    staged: dict[str, Any] = {}
    withheld: list[str] = []
    for name, value in items:
        if value is None or (isinstance(value, cp.ndarray)
                             and name not in _WITHHELD_TENDENCY_ARRAYS):
            staged[name] = value
        else:
            withheld.append(name)
    return _SubcycleTendencies(**staged, _withheld=tuple(sorted(withheld)))


class _Allocations:
    """Record pooled scratch addresses and reuse them during capture.

    Only addresses are recorded during the warm pass, so temporary arrays
    release their blocks for normal pool reuse. The private pool retains all
    those blocks for the graph's lifetime. Capture borrows the recorded
    addresses in the same allocation order without allocating CUDA memory.
    """

    def __init__(self, cp: Any) -> None:
        self.cp = cp
        self.allocator = cp.cuda.get_allocator()
        self.pool = cp.cuda.MemoryPool(self.allocator)
        self.device_id = cp.cuda.Device().id
        self.blocks: list[tuple[int, int]] = []
        self.replaying = False
        self.index = 0

    def __call__(self, size: int) -> Any:
        if not self.replaying:
            block = self.pool.malloc(size)
            self.blocks.append((size, int(block.ptr)))
            return block
        if self.index >= len(self.blocks) or self.blocks[self.index][0] != size:
            raise RuntimeError("subcycle scratch allocation sequence changed")
        pointer = self.blocks[self.index][1]
        self.index += 1
        return self.cp.cuda.MemoryPointer(self.cp.cuda.UnownedMemory(
            pointer, size, self.pool, device_id=self.device_id,
        ), 0)


class _Driving:
    def __init__(self, base: Any, tendencies: dict[str, Any]) -> None:
        self.base = base
        self.tendencies = tendencies

    def tendency(self, name: str) -> Any:
        return self.tendencies[name]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)


class SubcycleGraphs:
    """One reusable graph per driver, with eager fallback on layout changes."""

    def __init__(self, driver: Any) -> None:
        # Recovery imports its timing helper by value. This driver wrapper
        # delegates unchanged outside the context-local graph setup scope,
        # including calls from other threads and other drivers.
        from .cuda_backend import recovery
        recovery.launch_timed = launch_timed
        self.driver = driver
        self.cp = driver.cp
        self.graph = None
        self.key = None
        self.inputs = None
        self.result = None
        self.allocations = None
        self.tables = None
        self.last_launch = None
        self.require_replay = os.environ.get("WOOF_HEX_CUDA_GRAPHS_REQUIRE_REPLAY", "0") == "1"
        self.boundaries: dict[tuple[str, int], Any] = {}
        self.tendencies: dict[str, Any] = {}
        self.pool: dict[int, Any] = {}
        self.counter_delta = (0, 0, 0)
        self.disabled_key = None
        self.disabled_error = None
        self.subcycle = 0
        self.schedule = None
        self.fallbacks: list[dict[str, Any]] = []
        self.stats: dict[str, Any] = {
            "captured_subcycles": 0, "replayed_subcycles": 0,
            "eager_subcycles": 0, "fallback_reasons": {},
            "prepare_seconds": 0.0, "capture_seconds": 0.0,
        }

    def receipt(self) -> dict[str, Any]:
        return {**self.stats, "fallback_reasons": dict(self.stats["fallback_reasons"]),
                "fallbacks": [dict(row) for row in self.fallbacks],
                "retained_scratch_bytes": int(self.allocations.pool.total_bytes())
                if self.allocations is not None else 0,
                "descriptor_tables": None if self.tables is None else self.tables.receipt()}

    def _fallback(self, state: Any, reason: str, error: Any = None) -> None:
        regional = self.driver.regional_v841
        detail = self.disabled_error if error is None else _error_detail(error)
        if detail is None:
            detail = {"type": "CaptureNotAttempted", "message": reason, "site": []}
        row = {
            "subcycle": self.subcycle,
            "dynamics_subcycle": getattr(getattr(regional, "_offsets", None),
                                          "dynamics_substep", None),
            "model_time_seconds": float(state.time_seconds),
            "acoustic_subcycles": [{"rk_stage": stage.stage,
                                    "small_steps": list(range(1, stage.acoustic_steps + 1))}
                                   for stage in self.schedule.stages] if self.schedule is not None else [],
            "reason": reason, "capture_error": detail,
            "breakage_prevented": {
                "host_halo_exchange": "capture cannot perform host halo exchange",
                "garbage_audit_readback": "capture cannot perform host garbage audit readback",
                "external_capture": "nested stream capture is unsupported",
                "staging_memory_exhausted": "fixed graph inputs exceed available device memory",
                "scratch_memory_exhausted": "retained graph scratch exceeds available device memory",
                "prepare_memory_exhausted": "graph preparation exceeds available device memory",
                "physics_stage_callback": "a stage callback's host code runs only at capture, "
                                          "so replay would reuse its capture-time tendencies",
                "tendency_field_withheld": "the subcycle reads a tendency field the graph does "
                                           "not stage; replay would run without it (F1)",
            }.get(reason, "an incomplete capture cannot execute the admitted subcycle"),
        }
        self.fallbacks.append(row)
        counts = self.stats["fallback_reasons"]
        counts[reason] = counts.get(reason, 0) + 1
        from . import time_ledger
        time_ledger.emit("forecast.graph_fallback", attrs=row)
        if self.require_replay:
            raise RuntimeError(f"CUDA graph replay required; eager fallback: {reason}; {detail}")
        self.stats["eager_subcycles"] += 1

    def _eager(self, eager: Any, state: Any, kwargs: Any, reason: str, error: Any = None) -> Any:
        self._fallback(state, reason, error)
        return eager(state, **kwargs)

    def _counters(self) -> tuple[int, int, int]:
        regional = self.driver.regional_v841
        return (self.driver.mixing_calls_v841,
                0 if regional is None else regional.discipline.scrubs,
                0 if regional is None else getattr(regional.discipline, "fused_launches", 0))

    def _set_counters(self, value: tuple[int, int, int]) -> None:
        self.driver.mixing_calls_v841 = value[0]
        if self.driver.regional_v841 is not None:
            self.driver.regional_v841.discipline.scrubs = value[1]
            if hasattr(self.driver.regional_v841.discipline, "fused_launches"):
                self.driver.regional_v841.discipline.fused_launches = value[2]

    def _refresh_boundaries(self) -> None:
        regional = self.driver.regional_v841
        if regional is None:
            return
        for name in ("rho_zz", "rtheta_m", "ru"):
            self.tendencies[name] = _stage(regional.driving.tendency(name), self.cp,
                                         self.tendencies.get(name))
        for stage in (1, 2, 3):
            for name in ("rho_zz", "rtheta_m", "ru", "u"):
                key = (name, stage)
                self.boundaries[key] = regional.driving.state_at(
                    name, regional.clock, regional._stage_offset(stage),
                    out=self.boundaries.get(key),
                )

    @contextmanager
    def _boundary_scope(self):
        regional = self.driver.regional_v841
        if regional is None:
            yield
            return
        base = regional.driving
        old_hook = regional.driving_values
        regional.driving = _Driving(base, self.tendencies)
        regional.driving_values = lambda name, stage: self.boundaries[(name, stage)]
        try:
            yield
        finally:
            regional.driving = base
            regional.driving_values = old_hook

    def advance(self, eager: Any, state: Any, **kwargs: Any) -> Any:
        self.subcycle += 1
        self.schedule = kwargs.get("schedule")
        driver = self.driver
        cp = self.cp
        regional = driver.regional_v841
        if self.last_launch is not None:
            cp.cuda.get_current_stream().wait_event(self.last_launch)
        if driver.halo_exchanger_v841 is not None:
            return self._eager(eager, state, kwargs, "host_halo_exchange")
        if regional is not None and (
            regional.discipline.measure or regional.discipline.audit is not None
        ):
            return self._eager(eager, state, kwargs, "garbage_audit_readback")
        if cp.cuda.get_current_stream().is_capturing():
            return self._eager(eager, state, kwargs, "external_capture")
        if kwargs.get("physics_stage_callback") is not None:
            return self._eager(eager, state, kwargs, "physics_stage_callback")
        # F1 (2026-10-15): the graph sees the whole tendency carrier. An
        # allowlist of rho/rho_u/rho_theta dropped SG2's diabatic_theta, so
        # from step 2 every replay ran without MPAS's lagged microphysics
        # heating while eager added it (_SubcycleTendencies).
        staged_kwargs = dict(kwargs)
        tendencies = kwargs.get("physics_tendencies")
        if tendencies is not None:
            staged_kwargs["physics_tendencies"] = _subcycle_tendencies(tendencies, cp)
        dynamic = {"state": state, **staged_kwargs}
        atmosphere = driver.atmosphere
        key = (
            os.environ.get("WOOF_HEX_ACOUSTIC_FUSION", "0"),
            _layout(dynamic, cp), repr(kwargs["schedule"]), repr(driver.config),
            float(kwargs["outer_dt"]).hex(),
            tuple(_layout(getattr(atmosphere, name), cp, pointers=True)
                  for name in ("mesh", "vertical", "reference", "terrain")),
            _layout(driver.v841_context, cp, pointers=True),
            _layout(getattr(driver, "coefficients", None), cp, pointers=True),
            _layout(getattr(getattr(driver, "horizontal", None), "_deformation_v841", None),
                    cp, pointers=True),
            repr(getattr(driver, "mixing_config_v841", None)),
            id(regional), id(driver.cache.post_launch),
            None if regional is None else (
                _layout(regional.masks, cp, pointers=True),
                _layout(regional.mesh_scaling_cell, cp, pointers=True),
                _layout(regional.mesh_scaling_edge, cp, pointers=True),
                _layout(regional.edge_sign_on_vertex, cp, pointers=True),
                _layout(regional.cells_on_edge_sentinel, cp, pointers=True),
                _layout(regional.spec_zone_mask_cell, cp, pointers=True),
                regional.discipline.enabled,
                tuple(sorted(regional.discipline._immutable)),
                repr(regional.outer_dt), repr(regional.config_relax_zone_divdamp_coef),
            ),
        )
        if key != self.key:
            if self.graph is not None:
                self.last_launch.synchronize()
            self.graph = self.result = self.inputs = self.allocations = None
            self.tables = None
            self.boundaries.clear()
            self.tendencies.clear()
            self.pool.clear()
            self.key = key
            self.disabled_key = self.disabled_error = None
        if key == self.disabled_key:
            return self._eager(eager, state, kwargs, "capture_unavailable")
        try:
            self.inputs = _stage(dynamic, cp, self.inputs)
            self._refresh_boundaries()
        except cp.cuda.memory.OutOfMemoryError as error:
            cp.cuda.get_current_stream().synchronize()
            self.disabled_key = key
            self.graph = self.result = self.allocations = self.inputs = None
            self.tables = None
            self.boundaries.clear()
            self.tendencies.clear()
            self.pool.clear()
            self.disabled_error = _error_detail(error)
            return self._eager(eager, state, kwargs, "staging_memory_exhausted", error)
        # All work stays on the caller stream except the initial capture.
        # Capture is recorded after staging is complete and replay then joins
        # the same stream, so the eager flux reduction cannot race the graph.
        if self.graph is None:
            # Initialize lazy device constants and compile kernels outside the
            # recorded allocator. This preparation never commits a subcycle.
            # Restore caller inputs and counters before recording its launches.
            mark = time.perf_counter()
            before = self._counters()
            before_pool = {} if regional is None else dict(regional.discipline._unit_pool)
            preparation_error = None
            try:
                with graph_launch_scope(), self._boundary_scope():
                    prepared = eager(self.inputs["state"], **{
                        k: v for k, v in self.inputs.items() if k != "state"
                    })
                cp.cuda.get_current_stream().synchronize()
                del prepared
            except (cp.cuda.memory.OutOfMemoryError, WithheldTendencyRead) as error:
                preparation_error = error
            finally:
                self._set_counters(before)
                if regional is not None:
                    regional.discipline._unit_pool.clear()
                    regional.discipline._unit_pool.update(before_pool)
                self.stats["prepare_seconds"] += time.perf_counter() - mark
            if preparation_error is not None:
                cp.cuda.get_current_stream().synchronize()
                self.disabled_key = key
                self.disabled_error = _error_detail(preparation_error)
                self.inputs = None
                self.boundaries.clear()
                self.tendencies.clear()
                reason = ("tendency_field_withheld"
                          if isinstance(preparation_error, WithheldTendencyRead)
                          else "prepare_memory_exhausted")
                return self._eager(eager, state, kwargs, reason, preparation_error)
            self.inputs = _stage(dynamic, cp, self.inputs)
            mark = time.perf_counter()
            allocations = _Allocations(cp)
            table_owners = [] if regional is None else [
                getattr(regional, "solve_region", None),
                getattr(regional.discipline, "fused", None),
            ]
            tables = GraphTables(cp, table_owners, allocations.allocator)
            before = self._counters()
            before_pool = {} if regional is None else dict(regional.discipline._unit_pool)
            warm_memory_failure = False
            try:
                with cp.cuda.using_allocator(allocations), tables.scope(), graph_launch_scope(), self._boundary_scope():
                    warm = eager(self.inputs["state"], **{
                        k: v for k, v in self.inputs.items() if k != "state"
                    })
            except cp.cuda.memory.OutOfMemoryError as error:
                warm_memory_failure = True
                self.disabled_error = _error_detail(error)
            if warm_memory_failure:
                cp.cuda.get_current_stream().synchronize()
                self._set_counters(before)
                if regional is not None:
                    regional.discipline._unit_pool.clear()
                    regional.discipline._unit_pool.update(before_pool)
                allocations.blocks.clear()
                self.inputs = None
                allocations.pool.free_all_blocks()
                self.disabled_key = key
                self.stats["capture_seconds"] += time.perf_counter() - mark
                return self._eager(eager, state, kwargs, "scratch_memory_exhausted")
            after = self._counters()
            self.counter_delta = tuple(a - b for a, b in zip(after, before))
            if regional is not None:
                self.pool = dict(regional.discipline._unit_pool)
            cp.cuda.get_current_stream().synchronize()
            stream = cp.cuda.Stream(non_blocking=True)
            allocations.replaying = True
            tables.capturing = True
            if regional is not None:
                # A pooled address can be ordinary scratch early in the
                # subcycle and a recovered density later. Replay the same
                # registration order during capture, rather than classifying
                # that address by the warm pass's final density bindings.
                regional.discipline._unit_pool.clear()
                regional.discipline._unit_pool.update(before_pool)
            try:
                with stream, cp.cuda.using_allocator(allocations), tables.scope(), graph_launch_scope(), self._boundary_scope():
                    stream.begin_capture()
                    try:
                        captured = eager(self.inputs["state"], **{
                            k: v for k, v in self.inputs.items() if k != "state"
                        })
                    except BaseException as error:
                        # end_capture may report only capture invalidation.
                        # Preserve the body error and its concrete call site.
                        try:
                            stream.end_capture()
                        except BaseException as cleanup:
                            error.add_note(f"end_capture: {type(cleanup).__name__}: {cleanup}")
                        raise
                    else:
                        graph = stream.end_capture()
                if allocations.index != len(allocations.blocks):
                    raise RuntimeError("subcycle scratch allocation count changed")
            except (RuntimeError, cp.cuda.runtime.CUDARuntimeError) as error:
                self._set_counters(after)
                if regional is not None:
                    regional.discipline._unit_pool.clear()
                    regional.discipline._unit_pool.update(self.pool)
                self.pool.clear()
                self.disabled_key = key
                reason = f"capture_failed:{type(error).__name__}:{error}"
                self.disabled_error = _error_detail(error)
                self.stats["capture_seconds"] += time.perf_counter() - mark
                self._fallback(state, reason, error)
                # The warm pass already computed the staged subcycle exactly
                # once. Publish it instead of running an extra eager advance.
                cp.copyto(kwargs["validation_flag"], self.inputs["validation_flag"])
                self.inputs = None
                return warm
            self.graph, self.result, self.allocations = graph, captured, allocations
            self.tables = tables
            self.stats["capture_seconds"] += time.perf_counter() - mark
            self._set_counters(before)
            self.stats["captured_subcycles"] += 1
        before = self._counters()
        if regional is not None:
            regional.discipline._unit_pool.update(self.pool)
        self.graph.launch(stream=cp.cuda.get_current_stream())
        cp.copyto(kwargs["validation_flag"], self.inputs["validation_flag"])
        if self.last_launch is None:
            self.last_launch = cp.cuda.Event(disable_timing=True)
        self.last_launch.record(cp.cuda.get_current_stream())
        self._set_counters(tuple(a + b for a, b in zip(before, self.counter_delta)))
        self.stats["replayed_subcycles"] += 1
        # Transport rebinds scalars and the outer step advances time_seconds.
        # Fresh host containers keep those edits out of the retained graph.
        return replace(self.result, state=replace(
            self.result.state, time_seconds=state.time_seconds,
            scalars=self.inputs["state"].scalars,
        ))
