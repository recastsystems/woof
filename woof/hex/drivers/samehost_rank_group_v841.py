"""The ArWen-physics route on several cards: each rank's work as named operations, run by rank threads or by one
process per card (the default on Linux).

THE BREAKAGE THIS PREVENTS (woof.hex.rank_processes names the measurement): the ArWen-physics route ran every rank
as a thread of the forecast process, so on an ordinary CPython with the GIL every rank's kernel-launch work shared
one interpreter lock and more cards could be slower than fewer.  The MPAS-physics route moved to one process per
card on 2026-10-28 (8x RTX 5090: 8 cards about 7 s/step with threads, 0.145 s/step with processes); this module
gives the ArWen-physics route the same executor, so the two routes scale the same way on any Python.

How it is split.  ``samehost_forecast_v841._execute_samehost`` stays the one coordinator: it owns the clock, the
history and boundary writers, the receipts and every file.  Everything it used to do to ONE rank (build the stack,
step it, capture a history frame, read health, download a checkpoint) is an operation here, a function of that
rank's :class:`RankContext`.  The thread executor runs the operations on :class:`~woof.hex.partition_samehost_v841.SameHostRanks`
workers exactly as the coordinator's closures did; the process executor runs rank 0's in the forecast process and
ranks 1..N-1's in child processes (``woof.hex.rank_processes``), each child on its own card with its own
interpreter.  Both run the same operation code on the same inputs; halos move the same bytes in the same round
order (``partition_process_transport_v841``), so the words do not depend on the executor.

A child rebuilds the forecast's module state the way the forecast process built it (the door's mesh bind, the
driver's physics resolution) and then takes the forecast process's module constants (every upper-case data value
of the hexcore and woof modules), so a constant the host preparation rebound is the same in every rank.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from types import MappingProxyType, SimpleNamespace

import numpy as np

MODULE = "woof.hex.drivers.samehost_rank_group_v841"

#: Host keys a child never needs: the global preparation (each child gets its own slices), the f000 humidity the
#: coordinator alone completes frames with, and the fingerprint seal (a thread of the forecast process).
_COORDINATOR_ONLY_HOST_KEYS = ("prepared", "constructor_values", "gwdo_host", "f000_surface_diagnostics",
                               "f000_start_humidity", "prepared_fingerprint_seal")
_SLICED_HOST_KEYS = ("prepared", "constructor_values", "gwdo_host", "f000_surface_diagnostics")


# --------------------------------------------------------------------------------------------------------------
# One rank's objects
# --------------------------------------------------------------------------------------------------------------

class RankContext:
    """One rank's resident objects and the per-rank state the coordinator's closures used to hold.

    ``device`` is the CUDA ordinal this process uses for the rank (0 in a rank process, which sees only its own
    card); ``card`` is the forecast's own device number for the rank, used in receipts and timeline events.
    """

    def __init__(self, *, rank, device, card, proof, forecast, host, schedule, output_root, layouts, tables,
                 stream, physics_scope, cp):
        self.rank, self.device, self.card = int(rank), int(device), int(card)
        self.proof, self.forecast, self.host = proof, forecast, host
        self.schedule, self.output_root = schedule, output_root
        self.layouts, self.tables = layouts, tables
        self.layout, self.table = layouts[self.rank], tables[self.rank]
        self.stream, self.physics_scope, self.cp = stream, physics_scope, cp
        self.stack = self.cache = self.exchanger = self.transport = None
        self.previous = None        # the surface updates the next step's physics reads
        self.forcing = None         # the dynamics tendencies carrier the next step's GF reads
        self.iau_forcing = None
        self.iau_index = self.rank  # index of this rank in its IAU forcing (0 in a rank process)
        self.active_interval = False
        self.gc = None

    def device_scope(self):
        """The rank's card and compute stream (the scope SameHostRanks gives every rank operation)."""
        cp = self.cp
        return cp.cuda.Device(self.device), (self.stream if self.stream is not None else nullcontext())


def local_host(host, layout):
    """The rank's host inputs: the global preparation sliced to the layout's local columns (construct_rank's
    own slicing, unchanged)."""
    from woof.hex.partition_local_mesh_v841 import slice_prepared_host
    from woof.hex.drivers.samehost_forecast_v841 import _prepared_attribute_view

    local = dict(host)
    for name in _SLICED_HOST_KEYS:
        # PreparedCudaInputs has init=False derived fields. Its legacy
        # dataclass slicer can mutate the source on replace() failure.
        # A plain attribute view leaves the global preparation intact.
        item = _prepared_attribute_view(host[name]) if name == "prepared" else host[name]
        local[name] = slice_prepared_host(item, layout)
    local["constructor_values"] = dict(local["constructor_values"])
    local["constructor_values"]["n_columns"] = layout.n_local_cells + int(host.get("regional") is not None)
    return local


def attach_view(host):
    """What ``arwen_surface_update.attach_host`` reads of the global host, without the global arrays."""
    mesh = host["prepared"].mesh
    arrays = {}
    order = mesh.arrays.get("cell_order_original")
    if order is not None:
        arrays["cell_order_original"] = np.asarray(order)
    view = SimpleNamespace(mesh=SimpleNamespace(dimensions=dict(mesh.dimensions), arrays=arrays,
                                                attrs=dict(getattr(mesh, "attrs", {}) or {})))
    return {"prepared": view, "surface_init_path": host["surface_init_path"]}


def child_host(host, layout):
    """A rank process's host: its local slices plus every small global key, never a global array it does not
    read."""
    local = local_host(host, layout)
    for name in ("f000_start_humidity", "prepared_fingerprint_seal"):
        local.pop(name, None)
    return local


# --------------------------------------------------------------------------------------------------------------
# Operations: each runs on its rank's thread (thread executor) or in its rank's process (process executor)
# --------------------------------------------------------------------------------------------------------------

def op_construct(ctx, *, local, attach, arwen_checkout, stop_on_refusal, compile_cache_root, floor, admit_memory,
                 restart_entry, card_selection):
    """construct_rank: the rank's device stack on its card, in its own stream arena and physics scope."""
    from woof.hex.cuda_backend import KernelCache, require_cuda
    from woof.hex.drivers import samehost_restart_v841 as restart
    from woof.hex.drivers.samehost_forecast_v841 import rank_event

    cp, proof, forecast = ctx.cp, ctx.proof, ctx.forecast
    layout, device, rank = ctx.layout, ctx.device, ctx.rank
    rank_event(ctx.output_root, "prep.device_upload", "begin", rank, device,
               attrs={"local_cells": layout.n_local_cells, "card_selection": card_selection})
    memory = None
    with (cp.cuda.Device(device), ctx.stream if ctx.stream is not None else nullcontext(),
          ctx.physics_scope.rank(rank, device)):
        if device >= cp.cuda.runtime.getDeviceCount():
            raise ValueError(f"device {device} is not visible")
        # Keep compiler images across forecasts. CuPy's source/options/compiler
        # key includes every mesh dimension embedded into CUDA source.
        rank_cache = Path(compile_cache_root) / f"rank{rank}"
        rank_cache.mkdir(parents=True, exist_ok=True)
        capability = require_cuda(device_id=device, min_compute=(12, 0), cache_dir=rank_cache)
        cache = KernelCache(capability=capability, cache_dir=rank_cache)
        # Same-card rank checks need the SUM of resident rank footprints.
        # A whole-mesh door estimate is a lower bound, not a per-rank cap.
        if admit_memory:
            memory = {"device": ctx.card, **proof.gpu_memory_admission(cp, minimum=int(floor))}
        if local is None:
            local = local_host(ctx.host, layout)
        elif isinstance(local, str) and local == "context":
            local = ctx.host   # a rank process's host is already its own slices (child_host)
        try:
            stack = proof._construct_device_stack(host=local, cache=cache, arwen_checkout=arwen_checkout,
                                                   backend_builder=forecast._backend_builder(ctx.host, arwen_checkout),
                                                   physics_rollback_snapshot=bool(stop_on_refusal),
                                                   **(restart.constructor_kwargs(restart_entry)
                                                      if restart_entry is not None else {}))
            from woof.hex.arwen_surface_update import attach_host
            attach_host(stack["backend"], attach if attach is not None else ctx.host, ctx.schedule, ctx.output_root,
                        dt=float(forecast.DT_SECONDS), layout=layout)
            stack["restart_forcing"] = (restart.restore_forcing(restart_entry, stack)
                                         if restart_entry is not None else None)
            from woof.hex.partition_overlap_v841 import interior_edges
            edge_mask = interior_edges(local["prepared"].mesh.arrays["cellsOnEdge"],
                                       n_owned_cells=layout.n_owned_cells,
                                       n_owned_edges=layout.n_owned_edges)
            edge_extent = stack["driver"].horizontal.nedges
            if edge_extent < edge_mask.size:
                raise ValueError("local device has fewer edges than the partition; prevents incomplete damping")
            edge_mask = np.pad(edge_mask, (0, edge_extent - edge_mask.size))
            stack["interior_edge_mask"] = cp.asarray(edge_mask)
            # Halo work is disposable. Native field widths do not refresh
            # every outer K3 value, and those discarded values can exceed
            # health limits while the owned forecast remains unchanged.
            # Every global cell and edge is checked by exactly one rank.
            stack["owned_health_extents"] = {
                stack["driver"].horizontal.ncells: layout.n_owned_cells,
                edge_extent: layout.n_owned_edges,
            }
        finally:
            # Construction also enqueues device-table uploads. Finish
            # them before a failed constructor releases scoped owners.
            cp.cuda.get_current_stream().synchronize()
    rank_event(ctx.output_root, "prep.device_upload", "end", rank, device)
    ctx.stack, ctx.cache = stack, cache
    ctx.forcing = stack["restart_forcing"]
    return memory


def op_initial_surface(ctx, *, surface_accumulators):
    """The rank's initial surface-accumulator snapshot (``_initial_rank_surface_updates`` for one rank)."""
    from woof.hex.drivers.samehost_forecast_v841 import _initial_rank_surface_updates
    values = _initial_rank_surface_updates(ctx.proof, [ctx.stack], [ctx.device], surface_accumulators, ctx.cp,
                                           streams=None if ctx.stream is None else [ctx.stream])[0]
    ctx.previous = values
    return None


def op_describe(ctx):
    """What the coordinator reads of a rank to build the halo schedule and check the configuration."""
    driver = ctx.stack["driver"]
    state = driver.atmosphere.state
    return {"configuration_sha256": driver.configuration_sha256,
            "n_species": int(state.scalars.shape[0]),
            "n_levels": int(state.rho.shape[0]), "cell_width": int(state.rho.shape[1]),
            "has_forcing": ctx.forcing is not None}


def op_setup_exchange(ctx, *, physics_rounds, regional):
    """The rank's halo exchanger on its transport, bound to its driver.

    A limited-area rank exchanges its stage scalars before every scalar boundary adjust on EVERY rank
    (``_exchange_before_scalar_boundary_adjust``).  Before this was an operation the hook sat after the
    exchanger loop inside the IAU block, so it reached only the last rank and only with IAU: every other
    limited-area rank issued fewer D rounds than the regional round law requires.
    """
    from woof.hex.partition_samehost_v841 import SameHostHaloExchanger
    from woof.hex.drivers.samehost_forecast_v841 import _exchange_before_scalar_boundary_adjust

    exchanger = SameHostHaloExchanger(ctx.table, ctx.transport, expected_sequence=tuple(physics_rounds))
    driver = ctx.stack["driver"]
    driver.halo_exchanger_v841 = exchanger
    exchanger.interior_edge_mask = ctx.stack["interior_edge_mask"]
    if regional:
        _exchange_before_scalar_boundary_adjust(driver.regional_v841, exchanger)
    ctx.exchanger = exchanger
    return None


def op_iau(ctx, *, source, n_levels, cell_width):
    """A rank process's own IAU forcing: its local cells and edges of the increment the coordinator formed once."""
    from woof.hex.iau_native import iau_forcing_from_source
    forcing, receipt = iau_forcing_from_source(source, layouts=[ctx.layout], cp=ctx.cp, devices=[ctx.device],
                                               cell_widths=[int(cell_width)])
    ctx.iau_forcing, ctx.iau_index = forcing, 0
    return receipt


def op_collect_owned(ctx):
    from woof.hex.drivers.samehost_identity_v841 import collect_owned
    return collect_owned(ctx.layouts, ctx.rank, ctx.stack)


def op_capture(ctx, *, step, deferred):
    from woof.hex.drivers.samehost_forecast_v841 import _capture_rank
    return _capture_rank(ctx.proof, ctx.host, ctx.stack, step, ctx.schedule["labels"][step], deferred=deferred)


def op_first_call(ctx):
    from woof.hex import f000_first_call
    return f000_first_call.read_arwen_first_call(ctx.stack["backend"], ctx.cp)


def op_advance(ctx, *, step, refl_due, first, card_selection, last=False, expected_time=None):
    """One composite step of this rank, with its halo rounds; the step body of the coordinator's ``advance``."""
    from woof.hex.drivers.samehost_forecast_v841 import rank_event
    from woof.hex.partition_executor_v841 import ExecutorError

    if expected_time is not None:
        # The process executor's half of SameHostRanks.step's clock check: every rank starts the step at the
        # coordinator's committed clock.
        before = float(ctx.stack["driver"].atmosphere.state.time_seconds)
        if before != float(expected_time):
            raise ExecutorError(f"rank {ctx.rank} clock {before} before step {step}, coordinator {expected_time}")
    if first:
        rank_event(ctx.output_root, "forecast.integrate", "begin", ctx.rank, ctx.device,
                   attrs={"step": step, "card_selection": card_selection})
        ctx.active_interval = True
    stack, exchanger = ctx.stack, ctx.exchanger
    exchanger.begin_step(step)
    iau, index = ctx.iau_forcing, ctx.iau_index
    result = ctx.proof.execute_composite_step(driver=stack["driver"], backend=stack["backend"],
        scalar_names=ctx.host["scalar_names"], physics_geometry=stack["physics_geometry"],
        kernel_cache=stack["driver"].cache, previous_surface_updates=ctx.previous,
        dynamics_tendencies=ctx.forcing, refl_10cm_due=bool(refl_due),
        before_phase_two=exchanger.round_physics_candidate,
        **({} if iau is None else {"held_forcing": (
            lambda held, start, length, **context:
            iau.add(index, held, start, length, **context))}))
    atmosphere = stack["driver"].atmosphere
    exchanger.round_step_boundary(atmosphere.state, atmosphere.saved, result.dynamics_tendencies)
    exchanger.end_step()
    ctx.previous = result.committed.surface_updates
    ctx.forcing = result.dynamics_tendencies
    return {"driver": asdict(result.committed.receipt),
            "backend": dict(result.backend_receipt), "clamp_d2h": result.clamp_d2h.as_dict(),
            "recovery": result.recovery.receipt()}


def after_advance(ctx, *, step, last, failed=False):
    """After the rank's stream fence: the step's timeline mark, and the interval's end on the last step."""
    from woof.hex.drivers.samehost_forecast_v841 import rank_event
    rank_event(ctx.output_root, "forecast.integrate", "mark", ctx.rank, ctx.device, attrs={"step": step})
    if last or failed:
        rank_event(ctx.output_root, "forecast.integrate", "end", ctx.rank, ctx.device, attrs={"step": step})
        ctx.active_interval = False


def op_clock(ctx):
    return float(ctx.stack["driver"].atmosphere.state.time_seconds)


def op_health(ctx, *, step):
    return ctx.forecast.step_health_gate(ctx.stack, step, ctx.cp)


def op_compile_manifest(ctx):
    return ctx.cache.compile_manifest()


def op_checkpoint(ctx):
    from woof.hex.drivers import samehost_restart_v841 as restart
    return restart.download_rank(ctx.stack, ctx.layout, ctx.forcing)


def op_collect_forcing(ctx, *, time_seconds):
    carrier = ctx.forcing
    if carrier is None:
        return None
    if float(carrier.time_seconds) != float(time_seconds):
        raise RuntimeError("final dynamics forcing does not share the committed clock")
    return {name: ctx.proof._host_array(getattr(carrier, name), ctx.cp) for name in ("rthdynten", "rqvdynten")}


def op_final(ctx):
    """The rank's receipts at the end of the run (compile timing, stochastic state, its halo rounds)."""
    out = {"compile_timing": ctx.cache.timing_receipt() if ctx.cache is not None else None,
           "exchange": ctx.exchanger.receipt() if ctx.exchanger is not None else None}
    driver = None if ctx.stack is None else ctx.stack["driver"]
    if driver is not None and hasattr(driver, "stochastic"):
        out["stochastic"] = driver.stochastic.receipt()
    if ctx.iau_forcing is not None:
        out["iau"] = ctx.iau_forcing.receipt()
    return out


OPS = {
    "construct": op_construct, "initial_surface": op_initial_surface, "describe": op_describe,
    "setup_exchange": op_setup_exchange, "iau": op_iau, "collect_owned": op_collect_owned,
    "capture": op_capture, "first_call": op_first_call, "advance": op_advance, "clock": op_clock,
    "health": op_health, "compile_manifest": op_compile_manifest, "checkpoint": op_checkpoint,
    "collect_forcing": op_collect_forcing, "final": op_final,
}


# --------------------------------------------------------------------------------------------------------------
# Executor choice
# --------------------------------------------------------------------------------------------------------------

def executor_choice(requested, devices):
    """Which executor runs the ArWen-route ranks, and why (the MPAS route's rule, ``samehost_v842.executor_choice``).

    ``--samehost-ranks threads`` (or WOOF_HEX_SAMEHOST_RANKS=threads) keeps every rank a thread of this process.
    One process per card is the default wherever the host supports it.
    """
    from woof.hex.rank_processes import supported
    requested = (requested or os.environ.get("WOOF_HEX_SAMEHOST_RANKS") or "auto").strip().lower()
    if requested not in ("auto", "processes", "threads"):
        raise ValueError(f"unknown same-host rank executor {requested!r} (auto, processes or threads)")
    if len(tuple(devices)) < 2:
        return "threads", "one rank"
    if requested == "threads":
        return "threads", "requested"
    available, why = supported()
    if requested == "processes":
        if not available:
            raise ValueError(f"one process per card was requested but is not available here: {why}")
        return "processes", "requested"
    if available:
        return "processes", "default (one process per card)"
    print(f"woof hex: same-host ranks run as threads of one process: {why}", file=sys.stderr, flush=True)
    return "threads", why


def rank_host_parallelism(executor):
    if executor == "processes":
        return "one process per rank"
    gil = getattr(sys, "_is_gil_enabled", None)
    return "serialised (GIL)" if gil is None or gil() else "rank threads (free-threaded Python)"


# --------------------------------------------------------------------------------------------------------------
# The thread executor
# --------------------------------------------------------------------------------------------------------------

class ThreadRankGroup:
    """Every rank a thread of this process: the coordinator's closures, as operations on in-process contexts."""

    executor = "threads"

    def __init__(self, contexts, devices, *, cp, physics_scope, streams):
        self.contexts = list(contexts)
        self.devices = tuple(devices)
        self.cp, self.physics_scope, self.streams = cp, physics_scope, streams
        self.ranks = None
        self.time_seconds = None

    @property
    def failed(self):
        return self.ranks is not None and self.ranks.failed

    @property
    def sync_seconds(self):
        return list(self.ranks.sync_seconds) if self.ranks is not None else [0.0] * len(self.devices)

    @property
    def sync_calls(self):
        return list(self.ranks.sync_calls) if self.ranks is not None else [0] * len(self.devices)

    def construct(self, per_rank, setup_workers):
        from woof.hex.partition_samehost_v841 import SameHostRanks

        def construct_rank(rank):
            return OPS["construct"](self.contexts[rank], **per_rank[rank])
        # The existing workers already isolate CUDA device, stream and physics resources.
        # Same-card ranks retain serial admission of their summed resident footprint.
        if setup_workers > 1 and len(set(self.devices)) == len(self.devices):
            with SameHostRanks(self.devices, None, cp=self.cp, physics_scope=self.physics_scope,
                               streams=self.streams) as setup_ranks:
                return setup_ranks.call(construct_rank)
        return [construct_rank(rank) for rank in range(len(self.devices))]

    def initial_surface(self, surface_accumulators):
        from woof.hex.drivers.samehost_forecast_v841 import _initial_rank_surface_updates
        stacks = [ctx.stack for ctx in self.contexts]
        values = _initial_rank_surface_updates(self.contexts[0].proof, stacks, list(self.devices),
                                               surface_accumulators, self.cp, streams=self.streams)
        for ctx, value in zip(self.contexts, values):
            ctx.previous = value

    def local_call(self, name, **kwargs):
        """An operation that needs no rank worker (no device work), on every rank in order."""
        return [OPS[name](ctx, **kwargs) for ctx in self.contexts]

    def open(self, transport):
        from woof.hex.partition_samehost_v841 import SameHostRanks
        self.ranks = SameHostRanks(self.devices, transport, cp=self.cp, physics_scope=self.physics_scope,
                                   streams=self.streams)
        return self

    def call(self, name, **kwargs):
        operation = OPS[name]
        return self.ranks.call(lambda rank: operation(self.contexts[rank], **kwargs))

    def step(self, step, *, dt, last, **kwargs):
        def advance(rank):
            return OPS["advance"](self.contexts[rank], step=step, last=last, **kwargs)

        def completed(rank):
            after_advance(self.contexts[rank], step=step, last=last, failed=self.ranks.failed)
        advance.after_synchronize = completed
        results = self.ranks.step(advance, lambda rank: self.contexts[rank].stack["driver"].atmosphere.state.time_seconds,
                                  dt=dt)
        self.time_seconds = self.ranks.time_seconds
        return results

    def active_ranks(self):
        return [ctx.rank for ctx in self.contexts if ctx.active_interval]

    def end_intervals(self, step):
        from woof.hex.drivers.samehost_forecast_v841 import rank_event
        for ctx in self.contexts:
            # step() has drained every submitted worker. Include
            # ranks that completed before another rank failed.
            if ctx.active_interval:
                rank_event(ctx.output_root, "forecast.integrate", "end", ctx.rank, ctx.device,
                           attrs={"step": step, "failed": True})
                ctx.active_interval = False

    def final(self):
        return [OPS["final"](ctx) for ctx in self.contexts]

    def close(self):
        if self.ranks is not None:
            self.ranks.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, error, tb):
        try:
            self.close()
        except BaseException as close_error:
            if error is None:
                raise
            BaseException.add_note(error, f"same-host rank group close also failed: {type(close_error).__name__}: {close_error}")
        return False


# --------------------------------------------------------------------------------------------------------------
# The process executor
# --------------------------------------------------------------------------------------------------------------

class _Ready:
    """A rank process's captured frame, already on the host (the coordinator's PendingSnapshot interface)."""

    def __init__(self, arrays):
        self.arrays = arrays

    def result(self):
        return self.arrays


class RankProcessFailed(RuntimeError):
    """A rank process raised in an operation; its traceback is in the forecast log and in this message."""


def _module_constants():
    """Every upper-case data value of the loaded hexcore and woof modules: what host preparation and the mesh bind
    may have rebound, for a rank process to take after its own bind."""
    allowed = (int, float, complex, str, bytes, bool, type(None), np.generic)

    def plain(value, depth=0):
        if isinstance(value, allowed):
            return True
        if depth > 6:
            return False
        if isinstance(value, np.ndarray):
            return value.dtype != object and value.nbytes <= (1 << 22)
        if isinstance(value, (tuple, list, frozenset)):
            return all(plain(item, depth + 1) for item in value)
        if isinstance(value, (dict, MappingProxyType)):
            return all(plain(key, depth + 1) and plain(item, depth + 1) for key, item in value.items())
        return False

    out = {}
    for name, module in list(sys.modules.items()):
        if module is None or not (name.startswith("woof.hex") or name.startswith("woof")):
            continue
        values = {}
        for key, value in list(vars(module).items()):
            # Public upper-case names only: an underscore name is a module's private cache (a preflight set, a
            # device table), and this process's would tell a rank process it had done work it has not.
            if key.isupper() and not key.startswith("_") and plain(value):
                values[key] = value
        if values:
            out[name] = values
    return out


def _apply_module_constants(constants):
    """Set the forecast process's module constants in this process (each module imported first if needed)."""
    import importlib
    applied, failed = 0, []
    for name, values in constants.items():
        module = sys.modules.get(name)
        if module is None:
            try:
                module = importlib.import_module(name)
            except BaseException as error:   # a module this rank never needs (a door-only tool)
                failed.append(f"{name}: {type(error).__name__}: {error}")
                continue
        for key, value in values.items():
            current = getattr(module, key, _MISSING)
            if current is _MISSING or not _same(current, value):
                setattr(module, key, value)
                applied += 1
    return {"constants_set": applied, "modules": len(constants), "modules_not_imported": failed}


_MISSING = object()


def _same(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, np.ndarray):
        return left.shape == right.shape and left.dtype == right.dtype and bool(np.array_equal(left, right))
    try:
        return bool(left == right)
    except Exception:
        return False


def replay_spec(door_replay):
    """What a rank process needs to rebuild this process's module state: the door's request and driver argv
    (the mesh bind and the physics resolution are replayed), and every module constant as it is now."""
    from woof.hex import numerics
    if door_replay is None:
        raise ValueError("one process per card on the WOOF-physics route needs the door's request to rebuild "
                         "the forecast's module state in each rank process")
    return {"numerics": numerics.mode(), "repo": door_replay["repo"], "request": door_replay["request"],
            "driver_argv": list(door_replay["driver_argv"]), "constants": _module_constants(),
            # The engine tree a run names is put on sys.path when it is first imported
            # (cuda_arwen_physics_v841); a rank process imports from the same places.
            "sys_path": list(sys.path)}


def bootstrap(spec):
    """Rebuild the forecast process's module state here: numerics, the door's mesh bind, the driver's physics
    resolution, then every module constant the forecast process holds."""
    for entry in reversed(spec.get("sys_path") or ()):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    from woof.hex import numerics
    numerics.set_mode(spec["numerics"], source="same-host rank process request")
    from woof.hex import forecast_door as door
    binding, driver = door._load_drivers(spec["repo"])
    door._bind(binding, driver, spec["request"])
    args = driver.parse_args(spec["driver_argv"])
    driver.resolve_driver_physics(args)
    driver.resolve_physics_backend_row(args)
    record = _apply_module_constants(spec["constants"])
    return driver, record


class ProcessRankGroup:
    """Rank 0 in this process, ranks 1..N-1 as child processes; one command at a time, replies in rank order."""

    executor = "processes"

    def __init__(self, contexts0, devices, tables, *, job, timeout):
        from woof.hex.rank_processes import RankProcesses
        self.ctx = contexts0
        self.contexts = [contexts0]
        self.devices = tuple(int(device) for device in devices)
        self.size = len(self.devices)
        self.timeout = float(timeout)
        neighbours = {rank: [int(peer.peer) for peer in tables[rank].peers] for rank in range(self.size)}
        self.neighbours = neighbours
        self.group = RankProcesses(self.devices, neighbours, module=MODULE, timeout=self.timeout, job=str(job))
        self.failed = False
        self.failure = None
        self.time_seconds = None
        self.sync_seconds = [0.0] * self.size
        self.sync_calls = [0] * self.size
        self.records = []
        self.exit_codes = {}
        self.started = False

    # -- life cycle ----------------------------------------------------------------------------------------------

    def start(self, requests):
        """Start the children and hand each its request (module replay, layouts, tables, its rank's inputs)."""
        self.group.on_failure(self._child_failed)
        self.group.start()
        self.started = True
        self.halo_links = {peer: (self.group.halo_sockets[peer], self.group.halo_flags.pop(peer))
                           for peer in self.neighbours[0]}

        def send(rank):
            self.group.step_links[rank].send(requests[rank])
            requests[rank] = None   # the child holds its own copy now
        # Every child's request at once (Link.send copies with the GIL released), as run_process_ranks sends the
        # MPAS route's: one after another, 8 ranks' host slices took seconds each before any child could build.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(1, self.size - 1), thread_name_prefix="hex-rank-request") as pool:
            list(pool.map(send, range(1, self.size)))
        return {"rank_processes": [dict(rank=rank, pid=process.pid) for rank, process in self.group.processes.items()],
                "rank_process_parent_pid": os.getpid()}

    def _child_failed(self, reason):
        self.failed = True
        if self.failure is None:
            self.failure = reason
        transport = self.ctx.transport
        if transport is not None:
            try:
                transport.abort(reason)
            except Exception:
                pass
        for link in self.group.step_links.values():
            link.shutdown()

    def abort(self, reason):
        """Rank 0's own abort point (the coordinator thread): a child's failure need not end this process."""
        self.group.rank0_responded()
        self.failed = True
        if self.failure is None:
            self.failure = str(reason)
        if self.ctx.transport is not None:
            try:
                self.ctx.transport.abort(reason)
            except Exception:
                pass

    # -- one command ---------------------------------------------------------------------------------------------

    def _send(self, name, common, per_rank):
        for rank in range(1, self.size):
            kwargs = dict(common)
            if per_rank is not None:
                kwargs.update(per_rank[rank])
            self.group.step_links[rank].send(("call", name, kwargs))

    def _receive(self, ranks=None):
        values, errors = {}, []
        for rank in (range(1, self.size) if ranks is None else ranks):
            try:
                reply = self.group.step_links[rank].recv()
            except Exception as error:
                errors.append(f"rank {rank}: {type(error).__name__}: {error}")
                continue
            kind = reply[0]
            if kind == "ok":
                values[rank] = reply[1]
                seconds, calls = reply[2]
                self.sync_seconds[rank] = seconds
                self.sync_calls[rank] = calls
            else:
                errors.append(f"rank {rank}: {reply[1]}\n{reply[2]}")
        return values, errors

    def _local(self, name, kwargs, after=None):
        """Rank 0's operation in this process, in its card, stream and physics scope, then its stream fence."""
        ctx = self.ctx
        cp = ctx.cp
        device, stream = ctx.device_scope()
        scope = nullcontext() if name in _UNSCOPED else ctx.physics_scope.rank(ctx.rank, ctx.device)
        with device, stream, scope:
            result = OPS[name](ctx, **kwargs)
            began = time.perf_counter()
            try:
                if ctx.stream is not None:
                    ctx.stream.synchronize()
                else:
                    cp.cuda.get_current_stream().synchronize()
            finally:
                self.sync_seconds[0] += time.perf_counter() - began
                self.sync_calls[0] += 1
            if after is not None:
                after()
        return result

    def call(self, name, *, per_rank=None, after=None, **common):
        from woof.hex.partition_executor_v841 import ExecutorError
        if self.failed:
            raise ExecutorError(f"a failed rank group cannot resume ({self.failure})")
        self._send(name, common, per_rank)
        local_error = None
        result0 = None
        try:
            kwargs = dict(common)
            if per_rank is not None:
                kwargs.update(per_rank[0])
            result0 = self._local(name, kwargs, after)
        except BaseException as error:
            local_error = error
            self.abort(f"rank 0: {type(error).__name__}: {error}")
        values, errors = self._receive()
        if local_error is not None:
            for text in errors:
                BaseException.add_note(local_error, text)
            raise local_error
        if errors:
            self.abort(errors[0])
            raise RankProcessFailed(f"{name} failed on a rank process: " + "\n".join(errors))
        return [result0] + [values[rank] for rank in range(1, self.size)]

    # -- the coordinator's calls -------------------------------------------------------------------------------

    def construct(self, per_rank, setup_workers):
        return self.call("construct", per_rank=per_rank)

    def initial_surface(self, surface_accumulators):
        self.call("initial_surface", surface_accumulators=surface_accumulators)

    def local_call(self, name, **kwargs):
        return self.call(name, **kwargs)

    def open(self, transport=None):
        """The step loop's scope: an error stops every rank; the children stay up for final() and close()."""
        return _StepScope(self)

    def step(self, step, *, dt, last, **kwargs):
        """One shared step: every rank starts at the committed clock and must end one dt later (the clock law of
        SameHostRanks.step); a rank process returns its clock with its receipts."""
        from woof.hex.partition_executor_v841 import ExecutorError
        expected = (self.time_seconds if self.time_seconds is not None
                    else float(OPS["clock"](self.ctx)))
        replies = self.call("advance", step=step, expected_time=expected, last=last, **kwargs,
                            after=lambda: after_advance(self.ctx, step=step, last=last))
        results = [replies[0]] + [reply[0] for reply in replies[1:]]
        after = [float(OPS["clock"](self.ctx))] + [float(reply[1]) for reply in replies[1:]]
        target = float(expected) + float(dt)
        if any(value != target for value in after):
            self.abort("rank clocks disagree after a step")
            raise ExecutorError(f"rank clocks after a step {after}, expected {target}")
        self.time_seconds = target
        return results

    def active_ranks(self):
        return [0] if self.ctx.active_interval else []

    def end_intervals(self, step):
        from woof.hex.drivers.samehost_forecast_v841 import rank_event
        ctx = self.ctx
        if ctx.active_interval:
            rank_event(ctx.output_root, "forecast.integrate", "end", ctx.rank, ctx.device,
                       attrs={"step": step, "failed": True})
            ctx.active_interval = False

    def final(self):
        """Every rank's end-of-run receipts; a rank process that has stopped is named instead."""
        out = [OPS["final"](self.ctx)]
        alive = [rank for rank in range(1, self.size) if self.group.processes[rank].poll() is None]
        for rank in alive:
            try:
                self.group.step_links[rank].send(("call", "final", {}))
            except Exception:
                pass
        values, errors = self._receive(alive)
        for rank in range(1, self.size):
            out.append(values.get(rank, {"rank_process": "stopped before its final receipts", "errors": errors}))
        return out

    def close(self):
        if not self.started:
            return
        self.group.rank0_responded()
        transport = self.ctx.transport
        for rank in range(1, self.size):
            try:
                self.group.step_links[rank].send(("close",))
            except Exception:
                pass
        records = {}
        for rank in range(1, self.size):
            try:
                reply = self.group.step_links[rank].recv()
                if reply and reply[0] == "closed":
                    records[rank] = reply[1]
            except Exception:
                pass
        codes = self.group.join(60.0 if not self.failed else 30.0)
        if any(code != 0 for code in codes.values()):
            self.group.kill()
        self.group.close()
        if transport is not None:
            transport.close()
        from woof.hex.rank_processes import process_cpu_record
        self.records = [process_cpu_record(0)] + [records.get(rank, dict(rank=rank, missing=True))
                                                    for rank in range(1, self.size)]
        self.exit_codes = {str(rank): code for rank, code in codes.items()}
        self.started = False
        bad = {rank: code for rank, code in codes.items() if code != 0}
        if bad and not self.failed:
            raise RuntimeError(f"rank processes failed: {bad} ({self.group.failure}); their tracebacks are in the "
                               "forecast log")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, error, tb):
        if error is not None:
            self.abort(f"coordinator: {type(error).__name__}: {error}")
        try:
            self.close()
        except BaseException as close_error:
            if error is None:
                raise
            BaseException.add_note(error, f"rank processes close also failed: {type(close_error).__name__}: {close_error}")
        return False


class _StepScope:
    def __init__(self, group):
        self.group = group

    def __enter__(self):
        return self.group

    def __exit__(self, exc_type, error, tb):
        if error is not None:
            self.group.abort(f"coordinator: {type(error).__name__}: {error}")
        return False


# --------------------------------------------------------------------------------------------------------------
# A rank process
# --------------------------------------------------------------------------------------------------------------

def worker(job, rank, spec_text):
    """A rank process: rebuild the module state, build this rank's context, then serve the coordinator."""
    from woof.hex.rank_processes import (Link, install_parent_kernel_store, process_cpu_record,
                                        set_parent_death_signal)
    set_parent_death_signal()
    # Before the request arrives: the forecast process's warm kernel store (rank_processes).
    install_parent_kernel_store()
    rank = int(rank)
    spec = json.loads(spec_text)
    idle = float(os.environ.get("WOOF_HEX_SAMEHOST_RANK_IDLE_SECONDS", "86400"))
    link = Link(socket.socket(fileno=int(spec["step"])), timeout=idle, name=f"step-{rank}-0")
    # The output link is the MPAS route's writer channel; this route gathers frames on the step link.
    try:
        socket.socket(fileno=int(spec["output"])).close()
    except OSError:
        pass
    ctx = None
    code = 0
    try:
        request = link.recv()
        ctx = _child_context(rank, spec, request)
        while True:
            try:
                message = link.recv()
            except Exception:
                code = 1   # the forecast process closed this rank's link without a close: it stopped the run
                break
            if message[0] == "close":
                break
            _, name, kwargs = message
            try:
                result = _child_call(ctx, name, kwargs)
                link.send(("ok", result, (ctx.sync_seconds, ctx.sync_calls)))
            except BaseException as error:
                text = f"{type(error).__name__}: {error}"
                trace = traceback.format_exc()
                print(trace, file=sys.stderr, flush=True)
                if ctx.transport is not None:
                    try:
                        ctx.transport.abort(f"rank {rank}: {text}")
                    except Exception:
                        pass
                code = 1
                try:
                    link.send(("error", text, trace))
                except Exception:
                    break
    except BaseException as error:
        traceback.print_exc()
        print(json.dumps(dict(rank=rank, pid=os.getpid(), failed=f"{type(error).__name__}: {error}")),
              file=sys.stderr, flush=True)
        code = 1
    finally:
        record = process_cpu_record(rank)
        record.update(exit_code=code)
        if ctx is not None:
            if ctx.gc is not None:
                ctx.gc.exit()
            if ctx.transport is not None:
                try:
                    ctx.transport.close()
                except Exception:
                    pass
        try:
            link.send(("closed", record))
        except Exception:
            pass
        link.close()
        sys.stdout.flush()
        sys.stderr.flush()
    return code


def _child_context(rank, spec, request):
    """This rank process's context: the forecast's modules rebuilt, its card, stream, physics scope and halo end."""
    grant = os.environ.get("GPU_MUTEX_CARDS", "").replace(",", " ").split()
    card = int(request["card"])
    if grant and 0 <= card < len(grant):
        # This process sees one card; the queue's grant list is narrowed to it (rank_event reads it).
        os.environ["GPU_MUTEX_CARDS"] = grant[card]
    driver, replay = bootstrap(request["replay"])
    import cupy as cp
    from woof.hex.drivers.samehost_forecast_v841 import _StepLoopGarbage
    from woof.hex.drivers.samehost_physics_scope_v841 import scoped_physics_resources
    from woof.hex.partition_process_transport_v841 import ProcessHaloTransport

    cp.cuda.Device(0).use()
    stream = None
    if request["setup_stream"] == "rank":
        with cp.cuda.Device(0):
            stream = cp.cuda.Stream(non_blocking=True)
    scope_manager = scoped_physics_resources()
    scope = scope_manager.__enter__()
    ctx = RankContext(rank=rank, device=0, card=card, proof=driver.proof, forecast=driver, host=request["host"],
                      schedule=request["schedule"], output_root=request["output_root"], layouts=request["layouts"],
                      tables=request["tables"], stream=stream, physics_scope=scope, cp=cp)
    ctx.scope_manager = scope_manager
    ctx.replay_record = replay
    ctx.sync_seconds, ctx.sync_calls = 0.0, 0
    ctx.gc = _StepLoopGarbage()
    links = {int(peer): (socket.socket(fileno=int(sock_fd)), int(flag_fd))
             for peer, (sock_fd, flag_fd) in spec["halo"].items()}
    devices_here = [0] * len(request["tables"])
    with cp.cuda.Device(0):
        ctx.transport = ProcessHaloTransport(request["tables"], devices_here, rank, links, xp=cp,
                                             timeout=float(request["timeout"]),
                                             field_halo_indices=request["field_indices"])
    return ctx


#: Operations that touch no physics resource: run without the rank's physics scope, as the thread executor runs
#: them (the initial surface snapshot ran on plain worker threads; describe/setup_exchange on the coordinator).
_UNSCOPED = frozenset({"initial_surface", "describe", "setup_exchange", "clock"})


def _child_call(ctx, name, kwargs):
    cp = ctx.cp
    if name == "advance" and ctx.gc is not None and not ctx.gc._active:
        ctx.gc.enter()
    if name in ("clock", "describe", "setup_exchange"):
        return OPS[name](ctx, **kwargs)
    if name == "final":
        out = OPS["final"](ctx)
        out["transport"] = ctx.transport.receipt() if ctx.transport is not None else None
        out["physics_resource_scope"] = ctx.physics_scope.receipt() if hasattr(ctx.physics_scope, "receipt") else None
        out["module_replay"] = ctx.replay_record
        return out
    scope = nullcontext() if name in _UNSCOPED else ctx.physics_scope.rank(ctx.rank, ctx.device)
    with cp.cuda.Device(ctx.device), (ctx.stream if ctx.stream is not None else nullcontext()), scope:
        result = OPS[name](ctx, **kwargs)
        began = time.perf_counter()
        try:
            (ctx.stream if ctx.stream is not None else cp.cuda.get_current_stream()).synchronize()
        finally:
            ctx.sync_seconds += time.perf_counter() - began
            ctx.sync_calls += 1
        if name == "advance":
            step = int(kwargs["step"])
            after_advance(ctx, step=step, last=bool(kwargs.get("last", False)))
            ctx.gc.after_step(step)
            result = (result, OPS["clock"](ctx))
    if name == "capture":
        # The coordinator assembles frames from host copies; a deferred capture is finished here, after the fence.
        result = result.result() if hasattr(result, "result") else result
    return result


if __name__ == "__main__":
    raise SystemExit(worker(sys.argv[1], sys.argv[2], sys.argv[3]))
