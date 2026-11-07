"""Executable archived-parent -> standalone-child CUDA proof.

This is deliberately small enough for a rental smoke, but it runs the real
Morrison kernels, writes ordinary woof WRF-history files, destroys the live
parent, builds a specified child plus native coupled LBC tables from disk, and
advances that child with the production RK3/acoustic driver.  It is not an
online nest and never invokes WPS, real.exe, wrf.exe, or ndown.exe.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import fields as dataclass_fields, is_dataclass
from dataclasses import replace
from datetime import datetime, timedelta
import gc
import json
import os
from pathlib import Path
import time

import numpy as np

from woof.aerosol_source_receipt import aerosol_source_report_entry
from woof.config import soil_layer_count, validate_run_config
from woof.offline_child_run import (_memory_snapshot, _strict_json,
                                     child_health_log_fields)
from woof.offline_child import (
    OfflineChildPlacement,
    bind_parent_physics_from_gpuwm_restart,
    build_offline_child_domain_state,
    build_offline_lateral_boundaries,
    interpolate_parent_initial_state,
    validate_parent_history,
)


def _log(event: str, **values) -> None:
    # THE SAME STRICT WRITER the child door uses, for the reason that
    # door uses it: NaN is not a JSON token, so a line carrying one is
    # refused by every reader outside Python, and the lines this proof
    # writes most are the ones whose readings went.
    print(_strict_json({"event": event, **values}, sort_keys=True),
          flush=True)


def _device_allocation_inventory(root, cp) -> dict[str, object]:
    """Inventory unique CUDA allocations reachable from one model object.

    Views in the shared dycore workspace alias one allocation.  Reporting the
    base allocation once, with every reachable array name as an alias, keeps
    the byte total physical rather than double-counting workspace views.
    """

    allocations: dict[int, dict[str, object]] = {}
    seen_objects: set[int] = set()

    def visit(value, path: str, depth: int = 0) -> None:
        if value is None or depth > 8:
            return
        if isinstance(value, cp.ndarray):
            memory = value.data.mem
            pointer = int(memory.ptr)
            entry = allocations.setdefault(pointer, {
                "allocation_ptr": pointer,
                "allocation_bytes": int(memory.size),
                "arrays": [],
            })
            entry["arrays"].append({
                "name": path,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "view_bytes": int(value.nbytes),
            })
            return
        if isinstance(value, (str, bytes, bytearray, int, float, bool, np.generic)):
            return
        identity = id(value)
        if identity in seen_objects:
            return
        seen_objects.add(identity)
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(item, f"{path}[{key!r}]", depth + 1)
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]", depth + 1)
            return
        if is_dataclass(value) and not isinstance(value, type):
            for field in dataclass_fields(value):
                visit(getattr(value, field.name), f"{path}.{field.name}", depth + 1)
            return
        if value.__class__.__module__.startswith("woof") and hasattr(value, "__dict__"):
            for name, item in vars(value).items():
                visit(item, f"{path}.{name}", depth + 1)

    visit(root, "child")
    ordered = sorted(allocations.values(), key=lambda item: item["allocation_ptr"])
    return {
        "unique_allocations": len(ordered),
        "unique_allocation_bytes": sum(
            int(item["allocation_bytes"]) for item in ordered),
        "allocations": ordered,
    }


def _timing_summary(samples: list[float]) -> dict[str, object]:
    array = np.asarray(samples, dtype=np.float64)
    warm = array[1:] if array.size > 1 else array
    return {
        "count": int(array.size),
        "mean_seconds": float(array.mean()),
        "warm_mean_seconds": float(warm.mean()),
        "min_seconds": float(array.min()),
        "p50_seconds": float(np.percentile(array, 50.0)),
        "p95_seconds": float(np.percentile(array, 95.0)),
        "max_seconds": float(array.max()),
        "first_samples_seconds": array[:10].tolist(),
        "last_samples_seconds": array[-10:].tolist(),
    }


def _frame_fields(state, *, latitude: float = 35.0,
                  longitude: float = -97.0):
    import cupy as cp
    from woof.io.wrfout import state_frame

    ny, nx = state.mup.shape
    fields = state_frame(state, include_diagnostic_pressure=True)
    fields.update({
        "MAPFAC_M": cp.asnumpy(state.msft),
        "MAPFAC_U": cp.asnumpy(state.msfu),
        "MAPFAC_V": cp.asnumpy(state.msfv),
        "F": cp.asnumpy(state.f),
        "E": cp.asnumpy(state.e),
        "SINALPHA": cp.asnumpy(state.sina),
        "COSALPHA": cp.asnumpy(state.cosa),
        "XLAT": np.full((ny, nx), latitude, dtype=np.float32),
        "XLONG": np.full((ny, nx), longitude, dtype=np.float32),
    })
    return fields


def _write_single_frame(path: Path, state, cfg, valid_time: datetime) -> None:
    from woof.io.wrfout import WrfoutWriter

    global_attrs = {
        "MAP_PROJ": np.int32(1),
        "TRUELAT1": np.float32(30.0),
        "TRUELAT2": np.float32(60.0),
        "STAND_LON": np.float32(-97.0),
        "MOAD_CEN_LAT": np.float32(35.0),
        "CEN_LAT": np.float32(35.0),
        "CEN_LON": np.float32(-97.0),
        "POLE_LAT": np.float32(90.0),
        "POLE_LON": np.float32(0.0),
        "GRID_ID": np.int32(cfg.grid_id),
        "PARENT_ID": np.int32(0),
        "I_PARENT_START": np.int32(1),
        "J_PARENT_START": np.int32(1),
        "PARENT_GRID_RATIO": np.int32(1),
        "DT": np.float32(cfg.dt),
        "HYBRID_OPT": np.int32(cfg.hybrid_opt),
        "ETAC": np.float32(cfg.etac),
        "START_DATE": valid_time.strftime("%Y-%m-%d_%H:%M:%S"),
        "SIMULATION_START_DATE": valid_time.strftime("%Y-%m-%d_%H:%M:%S"),
    }
    with WrfoutWriter(
            path, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz,
            dx=cfg.dx, dy=cfg.dy,
            title="woof native offline-child parent archive",
            global_attrs=global_attrs,
            # The soil axis is the selected LSM's geometry, not a constant.
            soil_layers=soil_layer_count(cfg)) as writer:
        writer.write_frame(
            valid_time.strftime("%Y-%m-%d_%H:%M:%S"),
            _frame_fields(state))


def _parent_config(*, duration_seconds: float,
                   boundary_interval_seconds: float, mp_physics: int = 10):
    from woof.verify.cases import wk82

    return validate_run_config(replace(
        wk82.default_config(),
        nx=32, ny=32, nz=16, dx=3000.0, dy=3000.0,
        ztop=12000.0, dt=3.0, clock_dt=3.0,
        run_seconds=duration_seconds,
        output_interval_s=boundary_interval_seconds,
        moist=True, mp_physics=mp_physics,
        morr_rimed_ice=1, hybrid_opt=2, etac=0.2,
        hypsometric_opt=2, km_opt=1, khdif=0.0, kvdif=0.0,
        diff_6th_opt=0, diff_6th_slopeopt=0,
        damp_opt=0, w_damping=0, open_x=False, open_y=False,
        emdiv=0.0, specified=False, nested=False, terrain_opt=0,
        map_proj=0, sf_sfclay_physics=0, sf_surface_physics=0,
        bl_pbl_physics=0, ra_physics=0, cu_physics=0,
        time_step_sound=4, case=f"offline_child_parent_mp{mp_physics}", grid_id=3))


def _child_config(parent_cfg, *, duration_seconds: float,
                  output_interval_seconds: float):
    return validate_run_config(replace(
        parent_cfg,
        nx=24, ny=24, dx=1000.0, dy=1000.0,
        dt=1.0, clock_dt=1.0, run_seconds=duration_seconds,
        output_interval_s=output_interval_seconds,
        specified=True, nested=False,
        terrain_opt=1, map_proj=1, grid_id=4,
        case=f"offline_child_standalone_mp{parent_cfg.mp_physics}"))


def run(outdir: str | Path, *, duration_seconds: float = 9.0,
        boundary_interval_seconds: float = 3.0,
        output_interval_seconds: float | None = None,
        health_interval_seconds: float = 60.0,
        streaming_lbc: bool = False, mp_physics: int = 10,
        cloud_bump_kgkg: float = 0.0) -> dict[str, object]:
    import cupy as cp
    from woof.core.dycore import stability_report, step
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.physics import initialize_physics
    from woof.ingest.lateral_bc import (
        attach_lateral_boundaries, attach_streaming_lateral_boundaries,
        lateral_boundary_reload_count, lateral_boundary_resident_bytes)
    from woof.io.restart import restart_filename, write_restart
    from woof.io.wrfout import wrfout_filename
    from woof.verify.cases import wk82

    outdir = Path(outdir).resolve()
    parent_dir = outdir / "parent_archive"
    child_dir = outdir / "standalone_child"
    parent_dir.mkdir(parents=True, exist_ok=True)
    child_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    start_time = datetime(2000, 5, 20, 0, 0, 0)
    if duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be positive")
    if boundary_interval_seconds <= 0.0:
        raise ValueError("boundary_interval_seconds must be positive")
    if output_interval_seconds is None:
        output_interval_seconds = duration_seconds
    if output_interval_seconds <= 0.0 or health_interval_seconds <= 0.0:
        raise ValueError("output and health intervals must be positive")
    parent_cfg = _parent_config(
        duration_seconds=duration_seconds,
        boundary_interval_seconds=boundary_interval_seconds, mp_physics=mp_physics)
    if mp_physics == 40:
        parent_cfg = replace(parent_cfg, hailcast_opt=0)
    child_cfg = _child_config(
        parent_cfg, duration_seconds=duration_seconds,
        output_interval_seconds=output_interval_seconds)
    parent_steps = int(round(duration_seconds / parent_cfg.dt))
    child_steps = int(round(duration_seconds / child_cfg.dt))
    boundary_step_interval = int(round(
        boundary_interval_seconds / parent_cfg.dt))
    child_output_step_interval = int(round(
        output_interval_seconds / child_cfg.dt))
    child_health_step_interval = int(round(
        health_interval_seconds / child_cfg.dt))
    exact_requirements = {
        "duration/parent_dt": (duration_seconds / parent_cfg.dt, parent_steps),
        "duration/child_dt": (duration_seconds / child_cfg.dt, child_steps),
        "boundary_interval/parent_dt": (
            boundary_interval_seconds / parent_cfg.dt,
            boundary_step_interval),
        "output_interval/child_dt": (
            output_interval_seconds / child_cfg.dt,
            child_output_step_interval),
        "health_interval/child_dt": (
            health_interval_seconds / child_cfg.dt,
            child_health_step_interval),
    }
    for label, (raw, rounded) in exact_requirements.items():
        if rounded < 1 or not np.isclose(raw, rounded, rtol=0.0, atol=1e-9):
            raise ValueError(f"{label} must be a positive integer: {raw}")
    _log("parent_build_start", pid=os.getpid(), gpu=cp.cuda.runtime.getDevice())

    coord = make_vertical_coord(
        parent_cfg.nz, hybrid_opt=parent_cfg.hybrid_opt,
        etac=parent_cfg.etac)
    base = make_base_state(
        coord, lambda z: wk82.wk82_sounding(z)[0],
        p_surf=parent_cfg.p_surf, ztop=parent_cfg.ztop)
    parent = wk82.build(parent_cfg, coord, base)
    if cloud_bump_kgkg:
        if not 0.0 < cloud_bump_kgkg <= 1.0e-3:
            raise ValueError("cloud_bump_kgkg must be in (0, 0.001]")
        # Transfer vapour to cloud water in the lower cloud layer. This
        # makes number-carrier inheritance observable in the archive.
        parent.qc[3:parent_cfg.nz // 2] += cp.float32(cloud_bump_kgkg)
        parent.qv[3:parent_cfg.nz // 2] -= cp.float32(cloud_bump_kgkg)
    initialize_physics(parent, parent_cfg)
    parent_memory_initial = _memory_snapshot(cp)
    parent_pool_reserved_peak = parent_memory_initial["pool_reserved_bytes"]
    setup_path = write_restart(
        parent_dir / "gpuwmrst_d03_parent_setup.npz", parent, parent_cfg)
    binding = bind_parent_physics_from_gpuwm_restart(setup_path)

    parent_paths = []
    parent_step_seconds = []
    for step_index in range(parent_steps + 1):
        if step_index % boundary_step_interval == 0:
            frame_index = len(parent_paths)
            valid = start_time + timedelta(
                seconds=float(parent.elapsed_seconds))
            path = parent_dir / wrfout_filename(valid, domain_id=3)
            _write_single_frame(path, parent, parent_cfg, valid)
            parent_paths.append(path)
            health = stability_report(parent, parent_cfg)
            memory = _memory_snapshot(cp)
            parent_pool_reserved_peak = max(
                parent_pool_reserved_peak, memory["pool_reserved_bytes"])
            # THE SAME DECODER again: the parent's frame line took
            # ``float()`` to a w_max that is NaN exactly when the parent
            # went, and it is emitted immediately BEFORE the refusal
            # below, so the one line that reports a blown-up parent was
            # the one line no strict reader could open.
            _log("parent_frame", index=frame_index, path=str(path),
                 bytes=path.stat().st_size,
                 elapsed_seconds=float(parent.elapsed_seconds),
                 **child_health_log_fields(health),
                 memory=memory)
            if health["nan"]:
                raise RuntimeError(
                    f"parent became non-finite at step {step_index}")
        if step_index != parent_steps:
            step_started = time.perf_counter()
            step(parent, parent_cfg)
            cp.cuda.runtime.deviceSynchronize()
            parent_step_seconds.append(time.perf_counter() - step_started)
    parent_health = stability_report(parent, parent_cfg)
    del parent
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    free_after_parent, total_memory = cp.cuda.runtime.memGetInfo()
    _log("parent_destroyed", free_vram_bytes=int(free_after_parent),
         total_vram_bytes=int(total_memory))

    placement = OfflineChildPlacement(
        parent_nx=parent_cfg.nx, parent_ny=parent_cfg.ny,
        child_nx=child_cfg.nx, child_ny=child_cfg.ny,
        parent_grid_ratio=3, i_parent_start=10, j_parent_start=10)
    contract = validate_parent_history(
        parent_paths,
        max_boundary_interval_seconds=boundary_interval_seconds,
        physics_binding=binding)
    initial = interpolate_parent_initial_state(
        parent_paths[0], placement, physics_binding=binding,
        backend="cuda")
    boundary_result = build_offline_lateral_boundaries(
        contract, placement, backend="cuda",
        spec_bdy_width=child_cfg.spec_bdy_width,
        spec_zone=child_cfg.spec_zone, relax_zone=child_cfg.relax_zone)
    child = build_offline_child_domain_state(initial, child_cfg)
    attach = (attach_streaming_lateral_boundaries if streaming_lbc
              else attach_lateral_boundaries)
    attach(child, boundary_result.boundaries)
    boundary_device_resident_bytes = lateral_boundary_resident_bytes(child)
    initialize_physics(child, child_cfg)
    child_memory_initial = _memory_snapshot(cp)
    child_pool_reserved_peak = child_memory_initial["pool_reserved_bytes"]
    child_device_inventory = _device_allocation_inventory(child, cp)
    _log("child_launch", pid=os.getpid(), intervals=len(
        boundary_result.boundaries.intervals),
        initial_fields=len(initial.fields),
        initial_microphysics=len(initial.microphysics),
        memory=child_memory_initial,
        boundary_device_resident_bytes=boundary_device_resident_bytes,
        boundary_device_reload_count=lateral_boundary_reload_count(child),
        reachable_device_allocation_bytes=child_device_inventory[
            "unique_allocation_bytes"])

    child_initial_path = child_dir / wrfout_filename(start_time, domain_id=4)
    _write_single_frame(child_initial_path, child, child_cfg, start_time)
    child_output_paths = [child_initial_path]
    step_seconds = []
    last_health = stability_report(child, child_cfg)
    for step_index in range(1, child_steps + 1):
        step_started = time.perf_counter()
        step(child, child_cfg)
        cp.cuda.runtime.deviceSynchronize()
        step_seconds.append(time.perf_counter() - step_started)
        if (step_index % child_health_step_interval == 0 or
                step_index == child_steps):
            last_health = stability_report(child, child_cfg)
            memory = _memory_snapshot(cp)
            child_pool_reserved_peak = max(
                child_pool_reserved_peak, memory["pool_reserved_bytes"])
            # THE SAME DECODER the offline child's own loop uses, and
            # for the reason that loop states: ``cfl`` IS ``None``
            # whenever ``nan`` is true, and coercing it here raised
            # ``TypeError: float() argument must be a string or a real
            # number, not 'NoneType'`` in the health line immediately
            # before the refusal below (``offline_child_smoke.py`` lines
            # 388 and 393 of the 2.7.5 tree, the door's own pair being
            # lines 1460 and 1477).  So this proof died with that
            # traceback on exactly
            # the run it exists to report, which is the walked 2.7.5
            # failure with the door's name changed.
            _log("child_step", step=step_index,
                 total_steps=child_steps,
                 elapsed_seconds=float(child.elapsed_seconds),
                 **child_health_log_fields(last_health),
                 memory=memory,
                 wall_seconds=time.perf_counter() - started)
            if last_health["nan"]:
                raise RuntimeError(
                    f"offline child became non-finite at step {step_index}")
        if (step_index % child_output_step_interval == 0 or
                step_index == child_steps):
            output_time = start_time + timedelta(
                seconds=float(child.elapsed_seconds))
            output_path = child_dir / wrfout_filename(
                output_time, domain_id=4)
            _write_single_frame(output_path, child, child_cfg, output_time)
            if output_path not in child_output_paths:
                child_output_paths.append(output_path)
            _log("child_output", step=step_index,
                 elapsed_seconds=float(child.elapsed_seconds),
                 path=str(output_path), bytes=output_path.stat().st_size)
    child_final_path = child_output_paths[-1]
    # The instant name woof.resume.discover_checkpoint_sets recognises,
    # as the downscale runner writes it, so the smoke's child is a parent
    # the same way a downscaled run is.
    child_restart_path = write_restart(
        child_dir / restart_filename(
            start_time + timedelta(seconds=float(child.elapsed_seconds)),
            domain="d04"),
        child, child_cfg)
    child_health = stability_report(child, child_cfg)
    free_final, _ = cp.cuda.runtime.memGetInfo()
    report = {
        "result": "PASS" if not child_health["nan"] else "FAIL",
        "pipeline": f"gpuwm-mp{mp_physics}-parent-history-to-standalone-mp{mp_physics}-child",
        "mp_physics": mp_physics,
        "cloud_bump_kgkg": cloud_bump_kgkg,
        "online_parent_present_during_child": False,
        "parent_frames": [str(path) for path in parent_paths],
        "parent_frame_bytes": [path.stat().st_size for path in parent_paths],
        "parent_setup_restart": str(setup_path),
        "physics_binding": dict(binding.receipt()),
        "geometry_sha256": contract.geometry_sha256,
        "boundary_intervals": len(boundary_result.boundaries.intervals),
        "boundary_interval_seconds": contract.interval_seconds,
        "boundary_device_attachment": (
            "single-interval-streaming" if streaming_lbc else "all-eager"),
        "boundary_device_resident_bytes": boundary_device_resident_bytes,
        "boundary_device_reload_count": lateral_boundary_reload_count(child),
        "duration_seconds": duration_seconds,
        "parent_steps": len(parent_step_seconds),
        "parent_timing": _timing_summary(parent_step_seconds),
        "parent_memory_initial": parent_memory_initial,
        "parent_pool_reserved_peak_bytes": parent_pool_reserved_peak,
        "child_steps": len(step_seconds),
        "child_simulated_seconds": float(child.elapsed_seconds),
        "child_timing": _timing_summary(step_seconds),
        "child_memory_initial": child_memory_initial,
        "child_pool_reserved_peak_bytes": child_pool_reserved_peak,
        "child_device_inventory": child_device_inventory,
        "parent_health": parent_health,
        "child_health": child_health,
        "child_initial_wrfout": str(child_initial_path),
        "child_final_wrfout": str(child_final_path),
        "child_final_wrfout_bytes": child_final_path.stat().st_size,
        "child_output_paths": [str(path) for path in child_output_paths],
        "child_final_restart": str(child_restart_path),
        "child_final_restart_bytes": child_restart_path.stat().st_size,
        "free_vram_after_parent_destroy_bytes": int(free_after_parent),
        "free_vram_final_bytes": int(free_final),
        "total_vram_bytes": int(total_memory),
        "wall_seconds": time.perf_counter() - started,
    }
    # No aerosol dataset is read by either admitted fixed-droplet arm.
    # MP10 has no aerosol number carriers. MP40 leaves its activation
    # carriers inactive at aercu_opt=0, so neither arm has an aerosol
    # source to report.
    # It is called anyway -- rather than the decision being frozen into a
    # comment -- so that the day this smoke moves to an aerosol-aware
    # scheme its report gains the same block every other route's does,
    # instead of silently keeping the wrong shape.  Same inheritance
    # argument as woof/offline_child_run.py if it ever does: the child
    # takes QNWFA/QNIFA from the parent frames, it resolves no dataset.
    report.update(aerosol_source_report_entry(
        {}, mp_physics=child_cfg.mp_physics,
        when_unrecorded=(
            "this child's aerosol state would be interpolated from the "
            "archived parent history frames, not initialized from a "
            "dataset; the parent run's own report is where the source is "
            "recorded")))
    temporary = outdir / "report.json.tmp"
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, outdir / "report.json")
    _log("complete", **report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m woof.offline_child_smoke")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--mp-physics", type=int, choices=(10, 40), default=10)
    parser.add_argument("--cloud-bump-kgkg", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float, default=9.0)
    parser.add_argument("--boundary-interval-seconds", type=float,
                        default=3.0)
    parser.add_argument("--output-interval-seconds", type=float)
    parser.add_argument("--health-interval-seconds", type=float, default=60.0)
    parser.add_argument("--streaming-lbc", action="store_true")
    args = parser.parse_args(argv)
    report = run(
        args.outdir, duration_seconds=args.duration_seconds,
        boundary_interval_seconds=args.boundary_interval_seconds,
        output_interval_seconds=args.output_interval_seconds,
        health_interval_seconds=args.health_interval_seconds,
        streaming_lbc=args.streaming_lbc, mp_physics=args.mp_physics,
        cloud_bump_kgkg=args.cloud_bump_kgkg)
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
