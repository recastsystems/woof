"""Launch orchestration and continuation contract for NextGen components.

Original WOOF orchestration; no external hydrologic algorithm is implemented
here. Numerical work belongs to prepared CUDA/Rust components.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from types import MappingProxyType
from collections.abc import Mapping

_STATUS_OR = None


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


@dataclass(frozen=True)
class Stage:
    name: str
    component: object
    launch: object
    parallelism: str = "GPU"
    cadence_steps: int = 1
    interval_seconds: float | None = None


def _scheduled_stages(stages, schedule, interval):
    """Expand explicit whole-interval groups once, without advancing state."""
    if schedule is None:
        return stages, None
    if not isinstance(schedule, (list, tuple)) or not schedule:
        raise ValueError("after_schedule must contain explicit whole-interval groups")
    by_name = {stage.name: stage for stage in stages}
    used, expanded, identity = set(), [], []
    for group_index, group in enumerate(schedule):
        if not isinstance(group, Mapping) or set(group) != {"stages", "interval_seconds", "repeat"}:
            raise ValueError("schedule group needs stages, interval_seconds and repeat")
        names, dt, repeat = group["stages"], group["interval_seconds"], group["repeat"]
        if (not isinstance(names, (list, tuple)) or not names
                or any(not isinstance(name, str) or name not in by_name or name in used for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("schedule must name each after stage in exactly one group")
        if (type(dt) not in (int, float) or not math.isfinite(dt) or dt <= 0
                or type(repeat) is not int or repeat < 1 or dt * repeat != interval):
            raise ValueError("every schedule group must span exactly one parent interval")
        dt = float(dt)
        used.update(names)
        identity.append({"stages": list(names), "interval_seconds": dt, "repeat": repeat})
        for substep in range(repeat):
            for name in names:
                stage = by_name[name]
                if stage.cadence_steps != 1 or stage.interval_seconds is not None:
                    raise ValueError("subcycled stages cannot also have a multi-LSM cadence")

                def launch(parent_dt, *, stream=None, callback=stage.launch, step_dt=dt):
                    if parent_dt != interval:
                        raise ValueError("scheduled launch differs from its parent interval")
                    callback(step_dt, stream=stream)

                expanded.append(Stage(f"schedule{group_index}.{substep}.{name}",
                                      stage.component, launch, stage.parallelism))
    if used != set(by_name):
        raise ValueError("schedule omits after stages")
    return tuple(expanded), identity


class ResidentHydroDriver:
    """One member's fixed-parent state and prebound CUDA launch sequence.

    ``before`` stages precede WOOF's actual LSM call; ``after`` stages consume
    its runoff and advance the catchment/network state. Standalone binds its
    runoff modules in ``after`` as well. All preparation happens in factories.
    """

    def __init__(self, *, device_id, interval_seconds, identity,
                 before=(), after=(), resources=(), parent_grid_id=1, reach_output=None,
                 channel_output=None, status_fatal_masks=None, after_schedule=None,
                 stage_clocks=False, routing_clock=None):
        from woof.numerics import mode
        from .runtime import current_member_id
        self._member_id = current_member_id()
        self.device_id = int(device_id)
        self.parent_grid_id = int(parent_grid_id)
        self.interval_seconds = float(interval_seconds)
        if (self.parent_grid_id != 1 or not math.isfinite(self.interval_seconds)
                or self.interval_seconds <= 0):
            raise ValueError("hydro needs the fixed parent grid and a positive LSM interval")
        self._before = tuple(before)
        self._after = tuple(after)
        self.routing_clock = _json(routing_clock) if routing_clock else None
        if not self._after:
            raise ValueError("hydro has no bound runoff/routing stages")
        arrays, identities, components, statuses, component_masks = {}, {}, {}, {}, {}
        names = set()
        bindings = []
        for stage in (*self._before, *self._after):
            if not isinstance(stage, Stage) or not callable(stage.launch):
                raise ValueError("hydro stages need unique flat names and prebound launches")
            clocked = (stage_clocks and stage.interval_seconds is not None and stage.cadence_steps == 1
                       and not any(stage is bound for bound in self._before))
            if (type(stage.cadence_steps) is not int or stage.cadence_steps < 1
                    or (stage.interval_seconds is not None and not clocked
                        and stage.interval_seconds != self.interval_seconds * stage.cadence_steps
                        and not self.routing_clock)):
                raise ValueError("stage interval must span its whole number of parent LSM calls")
            if any(stage is bound for bound in self._before) and stage.cadence_steps != 1:
                raise ValueError("pre-LSM exchange must run at every LSM call")
            bindings.append((stage.name, stage.component))
        bindings.extend(resources)
        for name, component in bindings:
            if not isinstance(name, str) or not name or "/" in name or name in names:
                raise ValueError("hydro stages and resources need unique flat names")
            names.add(name)
            # The same component may own pre- and post-LSM launches.
            if id(component) in components:
                continue
            components[id(component)] = (name, component)
            carriers = component.arrays
            if not isinstance(carriers, Mapping):
                raise TypeError(f"{name}.arrays must be a mapping")
            identities[name] = _json(component.restart_identity)
            hooks = tuple(callable(getattr(component, hook, None)) for hook in
                          ("checkpoint_metadata", "validate_checkpoint", "restore_metadata"))
            if any(hooks) and not all(hooks):
                raise TypeError(f"{name} must provide all three continuation hooks")
            for key, array in carriers.items():
                if not isinstance(key, str) or not key or key.startswith("/"):
                    raise ValueError("hydro carrier names must be nonempty relative identifiers")
                interface = getattr(array, "__cuda_array_interface__", None)
                if (not isinstance(interface, dict)
                        or int(array.device.id) != self.device_id):
                    raise ValueError(f"{name}/{key} must reside on parent GPU {self.device_id}")
                if interface["data"][1]:
                    raise ValueError(f"{name}/{key} is read-only and cannot be restored")
                arrays[f"{name}/{key}"] = array
            diagnostics = getattr(component, "status_arrays", {})
            if not isinstance(diagnostics, Mapping):
                raise TypeError(f"{name}.status_arrays must be a mapping")
            for key, array in diagnostics.items():
                if (not isinstance(key, str) or not key or "/" in key
                        or str(array.dtype) not in ("int32", "uint32") or len(array.shape) != 1
                        or not array.size or not array.flags.c_contiguous
                        or array.device.id != self.device_id
                        or not any(array.data.ptr == carrier.data.ptr
                                   and array.nbytes == carrier.nbytes
                                   and carrier.dtype == array.dtype
                                   for carrier in carriers.values())):
                    raise ValueError(f"{name}/{key} status must be a checkpointed contiguous same-card int32 or uint32 vector")
                statuses[f"{name}/{key}"] = array
            policy = getattr(component, "status_fatal_masks", {})
            if (not isinstance(policy, Mapping)
                    or set(policy) - set(diagnostics)
                    or any(type(mask) is not int or not 0 <= mask <= 0xffffffff
                           for mask in policy.values())):
                raise ValueError(f"{name} status fatal masks must name bound diagnostics and unsigned 32-bit masks")
            component_masks.update({f"{name}/{key}": mask for key, mask in policy.items()})
        self.arrays = MappingProxyType(arrays)
        self.status_arrays = MappingProxyType(statuses)
        masks = component_masks | dict(status_fatal_masks or {})
        if (set(masks) - set(statuses)
                or any(type(value) is not int or not 0 <= value <= 0xffffffff
                       for value in masks.values())):
            raise ValueError("status fatal masks must name bound diagnostics and unsigned 32-bit masks")
        # Every bit is an error unless its owner explicitly declares advisory
        # bits. A realization can override that policy; never infer severity
        # from a component's name. Seal the effective policy in the checkpoint.
        self._status_fatal_masks = {key: masks.get(key, 0xffffffff) for key in statuses}
        self._components = tuple(components.values())
        self.components = MappingProxyType(dict(self._components))
        self._inventory = {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                           for key, value in arrays.items()}
        self._identity = _json({"schema": "arwen.nextgen.resident/v1",
                               "numerics": mode(),
                               "parent_grid_id": self.parent_grid_id,
                               "interval_seconds": self.interval_seconds,
                               "composition": identity,
                               "components": identities,
                               "before": [s.name for s in self._before],
                               "after": [s.name for s in self._after]})
        if self.member_id is not None:
            self._identity["member_id"] = self.member_id
        if masks:
            self._identity["status_fatal_masks"] = masks
        after_calls, schedule_identity = _scheduled_stages(
            self._after, after_schedule, self.interval_seconds)
        if schedule_identity is not None:
            self._identity["after_schedule"] = schedule_identity
        self._identity["stage_cadence"] = [
            {"name": s.name, "cadence_steps": s.cadence_steps,
             "interval_seconds": s.interval_seconds or self.interval_seconds}
            for s in after_calls]
        self._reach_output = None
        if reach_output is not None:
            ids, flow = reach_output["reach_id"], reach_output["discharge_m3_s"]
            if (str(ids.dtype) != "int64" or str(flow.dtype) not in ("float32", "float64")
                    or len(ids.shape) != 1 or ids.shape != flow.shape
                    or ids.device.id != self.device_id or flow.device.id != self.device_id
                    or not ids.flags.c_contiguous or not flow.flags.c_contiguous):
                raise ValueError("reach output needs ordered same-card contiguous int64 IDs and native-real discharge")
            # Products borrow these stable views at restart/output boundaries.
            # Unregistered publication storage would retain fresh-start values
            # after restoring the rest of the routing state.
            for array in (ids, flow):
                start, end = array.data.ptr, array.data.ptr + array.nbytes
                if not any(value.flags.c_contiguous and value.data.ptr <= start
                           and end <= value.data.ptr + value.nbytes for value in arrays.values()):
                    raise ValueError("reach output must be backed by checkpointed resident carriers")
            self._reach_output = MappingProxyType(dict(reach_output))
        self._channel_output = None
        if channel_output is not None:
            ids, depth = channel_output["reach_id"], channel_output["channel_depth_m"]
            if (str(ids.dtype) != "int64" or str(depth.dtype) not in ("float32", "float64")
                    or len(ids.shape) != 1 or not ids.size or ids.shape != depth.shape
                    or ids.device.id != self.device_id or depth.device.id != self.device_id
                    or not ids.flags.c_contiguous or not depth.flags.c_contiguous):
                raise ValueError("channel output needs ordered same-card contiguous int64 IDs and native-real above-bed depth")
            # Views into a registered publication buffer are valid. A separate
            # unregistered output would lose its contents on checkpoint restore.
            for array in (ids, depth):
                start, end = array.data.ptr, array.data.ptr + array.nbytes
                if not any(value.flags.c_contiguous and value.data.ptr <= start
                           and end <= value.data.ptr + value.nbytes for value in arrays.values()):
                    raise ValueError("channel output must be backed by checkpointed resident carriers")
            self._channel_output = MappingProxyType(dict(channel_output))
        self._before_calls = tuple(stage.launch for stage in self._before)
        self._after_calls = tuple((stage.launch, stage.cadence_steps,
                                   stage.interval_seconds or self.interval_seconds)
                                  for stage in after_calls)
        self.prepared_stages = (*self._before, *after_calls)
        self.before_stages = self._before
        # Timing adapters see exactly the expanded numerical callback order.
        # Checkpoint registration above retains original component namespaces.
        self.after_stages = after_calls
        self.steps = 0
        self.elapsed_seconds = 0.0
        self._pending = False
        # stage_clocks: a fixed LSM interval that does not divide a post-LSM
        # stage's period. Those stages then run on their own clock, launched at
        # the first LSM call at or after each multiple of their period (the
        # adaptive-parent schedule), so a forecast whose root step go lowered
        # below the prepared one keeps hydro on instead of dying at attach.
        # The routing clock (below) supersedes it for the provider's
        # exchange/runoff/routing/history rows, which it conserves across
        # boundaries the parent step does not divide.
        self.variable_interval = bool(stage_clocks)
        self.stage_launches = {stage.name: 0 for stage in self.prepared_stages}
        if self.routing_clock:
            if self._before or after_schedule or {s.name for s in self._after} != {'exchange', 'runoff', 'routing', 'history'}:
                raise ValueError('routing clock requires exchange, runoff, routing and history after the parent LSM')
            self._identity["routing_clock"] = self.routing_clock
            self._clock_period = float(self.routing_clock["period_seconds"])
            self._clock_duration = float(self.routing_clock["duration_seconds"])
            if not all(math.isfinite(v) and v > 0 for v in (self._clock_period, self._clock_duration)):
                raise ValueError('routing clock period and duration must be finite and positive')
            self._clock_calls = {s.name: s.launch for s in self._after}
            self._clock_exchange_time = 0.0
            self._clock_routed = 0
            self._clock_finalized = False

    def allow_variable_interval(self):
        """Follow an adaptive parent clock whose LSM interval changes every call.

        The routing clock samples instantaneous parents every call and accumulated
        parents on boundary crossings; both carry water across fixed routing dt.
        Legacy schedules must run on a fixed time interval of their own
        (``interval_seconds``) and is launched at the first LSM call at or after
        each multiple of it, with that interval; no stage may need every LSM
        call. Breakage it prevents (2.9.3 improve-1): the default product suite
        runs ``use_adaptive_time_step`` with ``bldt = 0``, so the parent LSM
        interval moves with the integration step and every coupled forecast
        stopped at the first LSM call with "hydro interval differs from the
        parent LSM cadence".
        """
        if self.routing_clock:
            self.variable_interval = True
            return
        if self._before:
            raise ValueError("an adaptive parent clock cannot drive pre-LSM hydro stages: "
                             + ", ".join(stage.name for stage in self._before))
        every_call = [stage.name for stage in self.after_stages if stage.interval_seconds is None]
        if every_call:
            raise ValueError("an adaptive parent clock changes the LSM interval every call; these "
                             "hydro stages need a fixed one: " + ", ".join(every_call))
        self.variable_interval = True

    @property
    def member_id(self):
        """Original ensemble member, or None outside a member capture scope."""
        return self._member_id

    @property
    def restart_identity(self):
        return _json(self._identity)

    def reach_output(self):
        if self._reach_output is None:
            raise RuntimeError("NextGen routing factory did not bind reach output")
        return self._reach_output

    def channel_output(self):
        """Cached optional MC-only axis; depth is metres above the channel bed.

        Its IDs may differ from the full network output, which can include
        reservoir water elevations. Consumers borrow these stable arrays on
        the producer stream at output cadence, before the next routing launch.
        """
        return self._channel_output

    def before_lsm(self, *, stream=None):
        if self._pending:
            raise RuntimeError("hydro pre-LSM sequence was already enqueued")
        self._pending = True
        for launch in self._before_calls:
            launch(self.interval_seconds, stream=stream)

    def advance(self, dt_seconds=None, *, stream=None):
        if self.routing_clock:
            return self._advance_clock(dt_seconds, stream=stream)
        if self.variable_interval:
            return self._advance_variable(dt_seconds, stream=stream)
        if dt_seconds is not None and dt_seconds != self.interval_seconds:
            raise ValueError("hydro interval differs from the parent LSM cadence")
        if not self._pending:
            raise RuntimeError("hydro advance requires its matching before_lsm call")
        for launch, cadence, interval in self._after_calls:
            if (self.steps + 1) % cadence == 0:
                launch(interval, stream=stream)
        self.steps += 1
        self.elapsed_seconds = self.steps * self.interval_seconds
        self._pending = False

    def _advance_clock(self, dt_seconds, *, stream=None):
        """Split resident runoff volumes at routing boundaries, never parent steps.

        Accumulated parents are sampled only on a crossing (or the last call).
        A sampled span is uniform in time; its excess volume stays in the next
        routing bucket. Instantaneous parents supply a depth on every call.
        All splitting and summation is in the runoff CUDA gather. Only clocks
        and launch decisions live here; no resident value is read by the host.
        """
        dt = self.interval_seconds if dt_seconds is None else float(dt_seconds)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("hydro needs a positive parent LSM interval")
        if not self.variable_interval and dt != self.interval_seconds:
            raise ValueError("hydro interval differs from the fixed parent LSM cadence")
        if not self._pending:
            raise RuntimeError("hydro advance requires its matching before_lsm call")
        if self._clock_finalized:
            raise RuntimeError("hydro routing clock already published its forecast end")
        start = self.elapsed_seconds
        end = start + dt if self.variable_interval else (self.steps + 1) * self.interval_seconds
        tol = self._clock_tolerance(end)
        final = end >= self._clock_duration - tol
        boundary = (self._clock_routed + 1) * self._clock_period
        accumulated = self.routing_clock["accumulated_parent"]
        sample = not accumulated or end >= boundary - tol or final
        if sample:
            span_start = self._clock_exchange_time if accumulated else start
            self._sample_clock(span_start, end, final=final, stream=stream)
        self.steps += 1
        self.elapsed_seconds = end
        self._clock_finalized = final
        self._pending = False

    def _clock_tolerance(self, end):
        """The most floating roundoff the running parent clock can hold, in seconds.

        ``elapsed_seconds`` is a sum of the parent's steps: each addition may round
        by half a unit in the last place, so the error grows with the step count.
        Breakage it prevents: 13,441 adaptive steps of exact tick fractions, 48 h in
        all, summed to 1.8e-8 s past 172,800 s, 38 times a fixed 16 ulp. That sliver
        opened a 577th routing step, the final capture fell outside the 48-frame
        history, and publication stopped on "history buffer exhausted", while a run
        whose sum ended below the end published. A real step is never this short:
        ticks are 1/900 s at their finest, against about 4e-7 s here.
        """
        return (16 + self.steps + 1) * math.ulp(max(end, self._clock_duration, self._clock_period))

    def _sample_clock(self, start, end, *, final, stream=None):
        span = end - start
        self._launch_clock("exchange", span, stream=stream)
        tol = self._clock_tolerance(end)
        cursor = start
        while cursor < end - tol:
            boundary = (self._clock_routed + 1) * self._clock_period
            stop = min(end, boundary)
            fraction = (stop - cursor) / span
            self._launch_clock("runoff", span, stream=stream, fraction=fraction,
                   reset=abs(cursor - self._clock_routed * self._clock_period) <= tol)
            cursor = stop
            terminal = final and cursor >= end - tol
            if cursor >= boundary - tol or terminal:
                self._launch_clock("routing", self._clock_period, stream=stream)
                # A forecast that ends on its boundary publishes the boundary: the
                # writer refuses a final time past its frame (nextgen_output series.rs).
                self._launch_clock("history", self._clock_period, stream=stream, final=terminal,
                       time_seconds=end if terminal and abs(end - boundary) > tol else boundary)
                self._clock_routed += 1
        self._clock_exchange_time = end

    def finish_clock(self, *, stream=None):
        """Flush a stopped forecast's partial bucket at the publication boundary.

        A fixed-step atmosphere may stop below its requested end time. Publish
        at the actual parent clock, consuming only water it really produced.
        This method enqueues CUDA work; it does not read a device value.
        """
        if not self.routing_clock or self._clock_finalized or not self.steps:
            return
        if self._pending:
            raise RuntimeError("cannot finish hydro between parent and routing launches")
        end = self.elapsed_seconds
        tol = self._clock_tolerance(end)
        routed_time = self._clock_routed * self._clock_period
        if self.routing_clock['accumulated_parent'] and self._clock_exchange_time < end - tol:
            self._sample_clock(self._clock_exchange_time, end, final=True, stream=stream)
        elif end > routed_time + tol:
            self._launch_clock("routing", self._clock_period, stream=stream)
            self._launch_clock("history", self._clock_period, stream=stream,
                               final=True, time_seconds=end)
            self._clock_routed += 1
        else:
            self._launch_clock("history", self._clock_period, stream=stream,
                               final=True, time_seconds=routed_time, recapture=True)
        self._clock_finalized = True

    def _launch_clock(self, name, span, *, stream=None, **kwargs):
        self._clock_calls[name](span, stream=stream, **kwargs)
        self.stage_launches[name] += 1

    def _advance_variable(self, dt_seconds, *, stream=None):
        dt = self.interval_seconds if dt_seconds is None else float(dt_seconds)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("hydro needs a positive parent LSM interval")
        if not self._pending:
            raise RuntimeError("hydro advance requires its matching before_lsm call")
        before, after = self.elapsed_seconds, self.elapsed_seconds + dt
        for launch, _, interval in self._after_calls:
            # Host arithmetic on the clock only; a millisecond tolerance keeps a
            # step that lands on the boundary from slipping to the next call.
            if math.floor((after + 1e-3) / interval) > math.floor((before + 1e-3) / interval):
                launch(interval, stream=stream)
        self.steps += 1
        self.elapsed_seconds = after
        self._pending = False

    def check_health(self, *, stream=None):
        """Inspect registered diagnostic bits at an output/checkpoint boundary.

        CUDA OR-reduces each vector, preserving all bits across rows; only the
        scalar masks cross to the host. This method never clears a carrier.
        Owners must retain sticky errors themselves between diagnostic reads.
        Call on the producer stream after the complete interval, never per step.
        Declared component policies retain advisory bits in the report without
        treating them as fatal; a realization may override them. Absent either
        declaration, every bit is fatal.
        """
        if self._pending:
            raise RuntimeError("cannot inspect hydro health between pre-LSM and routing launches")
        if not self.status_arrays:
            return {}
        import cupy as cp
        global _STATUS_OR
        if _STATUS_OR is None:
            # CuPy's bitwise_or ufunc does not implement reduce. Keep the
            # complete bit union on CUDA, including the signed i32 high bit.
            _STATUS_OR = cp.ReductionKernel(
                "int32 x", "uint32 y", "(unsigned int)x", "a | b", "y = a",
                "0", "nextgen_status_or")
        with cp.cuda.Device(self.device_id):
            stream = cp.cuda.get_current_stream() if stream is None else stream
            with stream:
                # Reinterpret unsigned carriers without copying or changing
                # their checkpoint dtype; the reduction preserves all bits.
                reduced = [_STATUS_OR(array.view(cp.int32))
                           for array in self.status_arrays.values()]
                masks = cp.stack(reduced).get(stream=stream)
        report = {key: int(value) for key, value in zip(self.status_arrays, masks)}
        failed = {key: value for key, value in report.items()
                  if value & self._status_fatal_masks[key]}
        if failed:
            error = RuntimeError(f"NextGen device diagnostics prevent publication: {failed}")
            error.diagnostics = report
            raise error
        return report

    def checkpoint_metadata(self):
        if self._pending:
            raise RuntimeError("cannot checkpoint between hydro pre-LSM and routing launches")
        diagnostics = self.check_health()
        continuation = {}
        for name, component in self._components:
            hook = getattr(component, "checkpoint_metadata", None)
            if hook is not None:
                continuation[name] = _json(hook())
        metadata = {"identity": self.restart_identity, "steps": self.steps,
                    "elapsed_seconds": self.elapsed_seconds,
                    "inventory": _json(self._inventory), "components": continuation}
        if self.routing_clock:
            metadata["routing_clock"] = {"exchange_time": self._clock_exchange_time,
                                         "routed": self._clock_routed,
                                         "finalized": self._clock_finalized,
                                         "launches": self.stage_launches}
        if diagnostics:
            metadata["diagnostics"] = diagnostics
        return metadata

    def validate_checkpoint(self, metadata, incoming):
        """Validate the complete payload before the atmosphere restores ANY array."""
        if not isinstance(metadata, dict) or metadata.get("identity") != self._identity:
            raise ValueError("NextGen checkpoint composition, hydrofabric or parameters differ")
        if metadata.get("inventory") != self._inventory or set(incoming) != set(self.arrays):
            raise ValueError("NextGen checkpoint omits or adds resident carriers")
        for key, target in self.arrays.items():
            value = incoming[key]
            if tuple(value.shape) != tuple(target.shape) or str(value.dtype) != str(target.dtype):
                raise ValueError(f"NextGen checkpoint shape/dtype differs for {key}")
        steps, elapsed = metadata.get("steps"), metadata.get("elapsed_seconds")
        if (type(steps) is not int or steps < 0 or type(elapsed) not in (int, float)
                or not math.isfinite(elapsed)
                or (elapsed != steps * self.interval_seconds
                    and not (self.variable_interval and 0 <= elapsed))):
            raise ValueError("NextGen checkpoint has inconsistent clocks")
        expected = {name for name, component in self._components
                    if callable(getattr(component, "checkpoint_metadata", None))}
        if not isinstance(metadata.get("components"), dict) or set(metadata["components"]) != expected:
            raise ValueError("NextGen checkpoint continuation metadata is incomplete")
        if self.routing_clock:
            clock = metadata.get("routing_clock", {})
            if (set(clock) != {"exchange_time", "routed", "finalized", "launches"}
                    or type(clock["routed"]) is not int or clock["routed"] < 0
                    or type(clock["finalized"]) is not bool
                    or not 0 <= clock["exchange_time"] <= elapsed
                    or set(clock["launches"]) != set(self.stage_launches)
                    or any(type(n) is not int or n < 0 for n in clock["launches"].values())):
                raise ValueError("NextGen checkpoint has inconsistent routing clock")
        for name, component in self._components:
            hook = getattr(component, "validate_checkpoint", None)
            if hook is not None:
                hook(metadata["components"][name])

    def restore_metadata(self, metadata):
        for name, component in self._components:
            hook = getattr(component, "restore_metadata", None)
            if hook is not None:
                hook(metadata["components"][name])
        self.steps = metadata["steps"]
        self.elapsed_seconds = metadata["elapsed_seconds"]
        if self.routing_clock:
            clock = metadata["routing_clock"]
            self._clock_exchange_time = clock["exchange_time"]
            self._clock_routed = clock["routed"]
            self._clock_finalized = clock["finalized"]
            self.stage_launches = dict(clock["launches"])
        self._pending = False
