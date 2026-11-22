"""Dry full-RK step timings and output hashes for one member-batched GPU.

This self-contained periodic workload uses the requested grid sizes. It is not
source-driven forecast qualification. Admission covers its explicit state,
scratch and glue backings plus a declared reserve. Timings exclude preparation,
compilation, warmup, output transfer and hashing. Sequential references occupy
one member at a time; no independent member processes share the card.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--nz", type=int, default=50)
    parser.add_argument("--dx", type=float, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--members", type=int, nargs="+", default=[1, 4, 10, 20, 40])
    parser.add_argument("--reserve-mib", type=int, default=512)
    parser.add_argument("--km-opt", type=int, choices=(1, 4), default=1)
    parser.add_argument("--diff6", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--layout-trial", choices=("none", "outermost", "innermost"), default="none")
    parser.add_argument("--advection-family", action="store_true")
    parser.add_argument("--terrain-height", type=float, default=0.0)
    parser.add_argument("--mapped", action="store_true")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--price-usd-per-hour", type=float)
    args = parser.parse_args()
    if min(args.nx, args.ny, args.nz, args.steps, *args.members) < 1 or args.warmup < 0 or args.reserve_mib < 0:
        parser.error("grid, steps and member counts must be positive; warmup/reserve nonnegative")
    if not np.isfinite(args.dx) or args.dx <= 0 or not np.isfinite(args.dt) or args.dt <= 0:
        parser.error("grid spacing and fixed timestep must be finite and positive")
    if not np.isfinite(args.terrain_height) or args.terrain_height < 0:
        parser.error("terrain height must be finite and nonnegative")
    if args.advection_family and args.layout_trial == "none":
        parser.error("advection family needs an explicit layout trial")
    if args.price_usd_per_hour is not None and (not np.isfinite(args.price_usd_per_hour) or args.price_usd_per_hour < 0):
        parser.error("hourly price must be finite and nonnegative")
    return args


def configuration(args):
    from woof.config import RunConfig
    return RunConfig(nx=args.nx, ny=args.ny, nz=args.nz, dx=args.dx, dy=args.dx,
                     ztop=12000.0, dt=args.dt,
                     run_seconds=(args.warmup + args.steps) * args.dt,
                     h_sca_adv_order=5, km_opt=getattr(args, "km_opt", 1),
                     diff_6th_opt=getattr(args, "diff6", 0),
                     terrain_opt=int(getattr(args, "terrain_height", 0.0) > 0))


def allocation_plan(cfg, reserve_bytes, *, has_msf=False):
    from woof.core.preflight import scratch_slot_registry
    from woof.ensemble.batch_diagnostics import diagnostics_specs
    from woof.ensemble.batch_glue import workspace_specs
    from woof.ensemble.batch_state import state_array_specs, SHARED_STATE_CANDIDATES
    from woof.ensemble.batch_storage import BatchArraySpec, BatchMemoryPlan
    from woof.core.device_inventory import state_array_shapes
    shared = tuple(sorted(SHARED_STATE_CANDIDATES & state_array_shapes(cfg).keys()))
    registry = scratch_slot_registry(cfg)
    slots = ("rk_ru", "rk_rv", "rk_ww", "acoustic_mu_pp_old", "acoustic_th_pp_old",
             "acoustic_c2a", "acoustic_a", "acoustic_alpha", "acoustic_gamma")
    extra = workspace_specs(cfg) + diagnostics_specs(cfg)
    if cfg.km_opt == 4 or cfg.diff_6th_opt > 0:
        from woof.ensemble import batch_mixing
        extra += batch_mixing.workspace_specs(cfg, has_msf=has_msf)
        slots += tuple(sorted(set(batch_mixing.required_scratch_slots(cfg)) - set(slots)))
    specs = state_array_specs(cfg, shared_fields=shared) + extra
    specs += tuple(BatchArraySpec("scratch:" + slot, registry[slot], "member") for slot in slots)
    return BatchMemoryPlan(specs, reserved_bytes=reserve_bytes), shared, extra, slots


def host_members(cfg, count, extras, *, terrain_height=0.0, mapped=False):
    """Prepare one static base and distinct member prognostics on the CPU."""
    from fractions import Fraction
    from woof.core.device_inventory import state_array_shapes
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import DomainState
    from woof.ensemble.batch_state import PreparedHostMember
    coord = make_vertical_coord(cfg.nz)
    terrain = None
    if terrain_height > 0:
        row, col = np.indices((cfg.ny, cfg.nx))
        terrain = terrain_height * (1.0 + 0.5 * np.sin(2 * np.pi * col / cfg.nx)
                                     * np.cos(2 * np.pi * row / cfg.ny))
    base = make_base_state(coord, lambda z: np.full_like(z, 300.0), cfg.p_surf, cfg.ztop,
                           terrain_z=terrain)
    template = DomainState(cfg, array_module=np)
    template.load_base(coord, base)
    if mapped:
        row, col = np.indices((cfg.ny, cfg.nx))
        urow, ucol = np.indices((cfg.ny, cfg.nx + 1))
        vrow, vcol = np.indices((cfg.ny + 1, cfg.nx))
        template.set_map_coriolis(
            msft=1.01 + 0.003 * np.sin(2 * np.pi * col / cfg.nx) * np.cos(2 * np.pi * row / cfg.ny),
            msfu=1.01 + 0.003 * np.sin(2 * np.pi * ucol / cfg.nx) * np.cos(2 * np.pi * urow / cfg.ny),
            msfv=1.01 + 0.003 * np.sin(2 * np.pi * vcol / cfg.nx) * np.cos(2 * np.pi * vrow / cfg.ny))
    names = state_array_shapes(cfg)
    initial = {name: getattr(template, name) for name in names}
    initial.update({spec.name: np.zeros(spec.shape, spec.dtype) for spec in extras})
    scalars = {name: value for name, value in vars(template).items()
               if name not in names and name not in {
                   "physics", "lateral_boundaries", "_scratch", "_scratch_arena",
                   "_host_setup_state", "_phb_host"}}
    rational = Fraction(str(cfg.dt))
    clock = {"ticks": 0, "step_ticks": rational.numerator, "tick_den": rational.denominator,
             "run_ticks": round(cfg.run_seconds * rational.denominator), "step_count": 0,
             "dt_fp32": np.float32(cfg.dt), "dtbc_fp32": np.float32(0)}
    result = []
    for member_index in range(count):
        seed = 2026100200 + member_index
        rng = np.random.default_rng(seed)
        arrays = dict(initial)
        for name, center, amplitude in (("u", 4.0, 0.02), ("v", -2.0, 0.02),
                                         ("thp", 0.0, 0.02), ("mup", 0.0, 0.2)):
            shape = names[name]
            phase = np.float32(rng.uniform(-np.pi, np.pi))
            row = np.arange(shape[-2], dtype=np.float32)[:, None]
            col = np.arange(shape[-1], dtype=np.float32)[None, :]
            pattern = (np.float32(center) + np.float32(amplitude) * np.sin(
                np.float32(2 * np.pi / cfg.nx) * col
                + np.float32(2 * np.pi / cfg.ny) * row + phase)).astype(np.float32)
            arrays[name] = np.broadcast_to(pattern, shape).copy()
        arrays["u"][..., -1] = arrays["u"][..., 0]
        arrays["v"][..., -1, :] = arrays["v"][..., 0, :]
        result.append(PreparedHostMember(cfg, arrays, scalars, dict(clock), phb_host=template._phb_host))
    return tuple(result)


def scalar_state(member):
    import cupy as cp
    from woof.core.state import DomainState
    from woof.core.device_inventory import state_array_shapes
    state = DomainState(member.cfg)
    for name in state_array_shapes(member.cfg):
        getattr(state, name).set(member.arrays[name])
    for name, value in member.scalars.items():
        setattr(state, name, value)
    state._phb_host = member.phb_host
    cp.cuda.get_current_stream().synchronize()
    return state


def state_hash(state, names):
    import cupy as cp
    result = hashlib.sha256()
    for name in names:
        array = cp.asnumpy(getattr(state, name))
        if not np.isfinite(array).all():
            raise RuntimeError("nonfinite output in " + name)
        result.update(name.encode("ascii"))
        result.update(array.dtype.str.encode("ascii"))
        result.update(np.array(array.shape, np.int64).tobytes())
        result.update(array.tobytes())
    return result.hexdigest()


def time_steps(step, steps):
    import cupy as cp
    start, end = cp.cuda.Event(), cp.cuda.Event()
    cp.cuda.get_current_stream().synchronize()
    wall_start = time.perf_counter()
    start.record()
    for _ in range(steps):
        step()
    end.record()
    cp.cuda.get_current_stream().synchronize()
    return time.perf_counter() - wall_start, cp.cuda.get_elapsed_time(start, end) * 0.001


def run(args):
    cfg = configuration(args)
    plan, shared, extras, slots = allocation_plan(cfg, args.reserve_mib * 1024**2,
                                                has_msf=getattr(args, "mapped", False))
    from woof.ensemble.batch_layout_trials import (
        layout_trial_workspace_bytes, layout_trial_workspace_payload_bytes)
    trial_selection = getattr(args, "layout_trial", "none")
    def required_bytes(count):
        return plan.required_bytes(count) + layout_trial_workspace_bytes(cfg, count, trial_selection)
    receipt = {"schema": "ensemble-dry-rk-step-probe-v1", "workload": vars(args).copy(),
               "scope": "periodic dry acoustic RK3; no source-driven weather skill or history timing",
               "allocation_scope": "explicit state/scratch/glue plus declared reserve; not whole forecast admission",
               "rows": []}
    receipt["workload"]["receipt"] = str(args.receipt)
    if args.estimate_only:
        receipt["rows"] = [{"members": n, "required_bytes": required_bytes(n),
                             "additional_trial_workspace_bytes": layout_trial_workspace_bytes(cfg, n, trial_selection),
                             "additional_trial_workspace_payload_bytes": layout_trial_workspace_payload_bytes(cfg, n, trial_selection)}
                            for n in args.members]
        return receipt
    import cupy as cp
    from woof.core import dycore
    from woof.core.device_inventory import state_array_shapes
    from woof.ensemble.batch_dycore import prepare_dry_step, member_domain_view
    from woof.ensemble.batch_state import BatchedDomainState
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    name = props["name"]
    receipt["gpu"] = name.decode() if isinstance(name, bytes) else name
    receipt["cuda_runtime"] = cp.cuda.runtime.runtimeGetVersion()
    receipt["cupy"] = cp.__version__
    names = tuple(sorted(state_array_shapes(cfg)))
    pool = cp.get_default_memory_pool()
    for count in args.members:
        gc.collect()
        pool.free_all_blocks()
        available, total = cp.cuda.runtime.memGetInfo()
        trial_bytes = layout_trial_workspace_bytes(cfg, count, trial_selection)
        fits = [n for n in range(1, 51) if required_bytes(n) <= available]
        row = {"members": count, "required_bytes": required_bytes(count),
               "additional_trial_workspace_bytes": trial_bytes,
               "additional_trial_workspace_payload_bytes": layout_trial_workspace_payload_bytes(cfg, count, trial_selection),
               "available_bytes": available, "total_bytes": total,
               "largest_that_fits": max(fits, default=0)}
        try:
            plan.admit(count, available_bytes=available - trial_bytes)
        except MemoryError as error:
            row.update(status="refused", reason=str(error))
            receipt["rows"].append(row)
            print(json.dumps(row), flush=True)
            continue
        inputs = host_members(cfg, count, extras,
                              terrain_height=getattr(args, "terrain_height", 0.0),
                              mapped=getattr(args, "mapped", False))
        batch = BatchedDomainState.from_prepared(inputs, array_module=cp, available_bytes=available - trial_bytes,
                    shared_fields=shared, extra_specs=extras, scratch_slots={slot: np.float32 for slot in slots},
                    reserved_bytes=plan.reserved_bytes)
        if batch.plan.required_bytes(count) != plan.required_bytes(count):
            raise RuntimeError("allocated inventory differs from the quoted inventory")
        step = prepare_dry_step(batch, layout_trial=trial_selection,
                                advection_family=getattr(args, "advection_family", False))
        row["layout_trial"] = step.trial_receipt
        for _ in range(args.warmup):
            step()
        planned_pool_bytes = plan.required_bytes(count) - plan.reserved_bytes + trial_bytes
        steady_pool_bytes = pool.used_bytes()
        if steady_pool_bytes != planned_pool_bytes:
            raise RuntimeError(
                f"warmed pool inventory differs from admission: {steady_pool_bytes} != {planned_pool_bytes}")
        batch_wall, batch_device = time_steps(step, args.steps)
        row["batch_pool_live_bytes"] = pool.used_bytes()
        row["planned_pool_live_bytes"] = planned_pool_bytes
        if row["batch_pool_live_bytes"] != steady_pool_bytes:
            raise RuntimeError("timed steps changed live CUDA allocation bytes")
        hashes = [state_hash(member_domain_view(batch, member), names) for member in range(count)]
        del step, batch
        gc.collect()
        pool.free_all_blocks()
        sequential_wall = sequential_device = 0.0
        for member, expected in zip(inputs, hashes, strict=True):
            scalar = scalar_state(member)
            for _ in range(args.warmup):
                dycore.step(scalar, cfg, acoustic=True)
            wall, device = time_steps(lambda: dycore.step(scalar, cfg, acoustic=True), args.steps)
            sequential_wall += wall
            sequential_device += device
            actual = state_hash(scalar, names)
            if actual != expected:
                raise RuntimeError("complete output identity failed: " + actual + " != " + expected)
            scalar = None  # rebound, not deleted: a closure above reads it
            gc.collect()
            pool.free_all_blocks()
        row.update(status="measured", identity="byte_identical_all_state_fields", output_sha256=hashes,
                   batch_wall_seconds=batch_wall, batch_device_seconds=batch_device,
                   sequential_wall_seconds=sequential_wall, sequential_device_seconds=sequential_device,
                   sequential_over_batch=sequential_wall / batch_wall,
                   simulated_seconds=args.steps * cfg.dt,
                   members_per_gpu_hour=count * 3600 / batch_wall)
        if args.price_usd_per_hour is not None:
            row["cost_usd_per_member"] = batch_wall * args.price_usd_per_hour / (3600 * count)
        receipt["rows"].append(row)
        print(json.dumps(row), flush=True)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        del inputs
    return receipt


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    args = arguments()
    result = run(args)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
