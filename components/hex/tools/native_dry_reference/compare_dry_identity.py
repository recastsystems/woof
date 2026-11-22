#!/usr/bin/env python3
"""Measure the hex dry dycore's numerical difference from native MPAS-A.

The native side is a directory of ``tracer.<time>.nc`` files written after every
model step by ``run_native_jw.sh``.  The hex side starts from the native file at
time zero and runs the same schedule.  Three comparisons are made:

* ``cpu``: the NumPy port, one step from each native output to the next;
* ``cuda``: the CUDA driver, one continuous run from native time zero;
* ``cpu_against_cuda``: step one of each, against the other.

For every field of every step the report gives: identical or not, how many
32-bit words differ, the largest absolute difference, the largest distance in
units in the last place, and the level, column and two values of the first
differing word.  It measures and refuses nothing; there is no tolerance in it.

Twelve fields are compared.  Nine are native arrays read straight from the
file (rho_zz, ru, rw, the scalars, theta_m, exner, rho_p, rtheta_p,
pressure_p) and two more are the recovered velocities (u, w).  ``rho_theta``
is the product rho_zz * theta_m formed by the loader, so a difference there is
not a native array's and is marked ``derived``.

``--package`` names the import package, so the same file runs against a tree
from before the 2026-08-28 rename (``mpas_port``).
"""

from __future__ import annotations
import os as _hex_os
_hex_os.environ["WOOF_HEX_NUMERICS"] = "strict"  # a bitwise proof against native MPAS runs strict numerics (fast is the default, strict the opt-in)

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np

STATE_FIELDS = ("rho", "rho_theta", "rho_u", "rho_w", "scalars")
SIDECAR_FIELDS = (
    "theta_m",
    "exner",
    "density_perturbation",
    "rho_theta_perturbation",
    "pressure_perturbation",
    "normal_velocity",
    "vertical_velocity",
)
NATIVE_NAME = {
    "rho": "rho_zz",
    "rho_theta": "rho_zz*theta_m (derived)",
    "rho_u": "ru",
    "rho_w": "rw",
    "scalars": "scalars (qv)",
    "theta_m": "theta_m",
    "exner": "exner",
    "density_perturbation": "rho_p",
    "rho_theta_perturbation": "rtheta_p",
    "pressure_perturbation": "pressure_p",
    "normal_velocity": "u",
    "vertical_velocity": "w",
}


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance in representable binary32 values; the two zeros are one value."""

    def ordered(x: np.ndarray) -> np.ndarray:
        bits = np.ascontiguousarray(x, dtype=np.float32).view(np.int32).astype(np.int64)
        return np.where(bits < 0, -(bits & 0x7FFFFFFF), bits)

    return np.abs(ordered(a) - ordered(b))


def field_report(actual: Any, expected: Any) -> dict[str, Any]:
    actual = np.ascontiguousarray(np.asarray(actual))
    expected = np.ascontiguousarray(np.asarray(expected))
    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.dtype != np.float32:
        return {"identical": False,
                "why": f"{actual.shape}/{actual.dtype} against {expected.shape}/{expected.dtype}"}
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return {"identical": False, "finite": False, "why": "non-finite values"}
    if actual.tobytes() == expected.tobytes():
        return {"identical": True, "finite": True, "words": int(actual.size),
                "different_words": 0, "max_abs": 0.0, "max_ulp": 0}
    differs = actual.view(np.uint32) != expected.view(np.uint32)
    gap = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    distance = ulp_distance(actual, expected)
    first = tuple(int(i) for i in np.argwhere(differs)[0])
    worst = tuple(int(i) for i in np.unravel_index(int(np.argmax(distance)), distance.shape))
    return {
        "identical": False,
        "words": int(actual.size),
        "different_words": int(np.count_nonzero(differs)),
        "max_abs": float(gap.max()),
        "max_ulp": int(distance.max()),
        "words_off_by_more_than_one_ulp": int(np.count_nonzero(distance > 1)),
        "first_index": first,
        "first_hex": float(actual[first]),
        "first_native": float(expected[first]),
        "worst_index": worst,
        "worst_hex": float(actual[worst]),
        "worst_native": float(expected[worst]),
    }


def compare(state: Any, saved: Any, native_state: Any, native_saved: Any) -> dict[str, Any]:
    report = {name: field_report(getattr(state, name), getattr(native_state, name))
              for name in STATE_FIELDS}
    report.update({name: field_report(getattr(saved, name), getattr(native_saved, name))
                   for name in SIDECAR_FIELDS})
    return report


def first_difference(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        for name in STATE_FIELDS + SIDECAR_FIELDS:
            entry = row["fields"][name]
            if not entry["identical"]:
                return {"step": row["step"], "field": name, "native_array": NATIVE_NAME[name],
                        **entry}
    return None


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    identical = [name for name in STATE_FIELDS + SIDECAR_FIELDS
                 if all(row["fields"][name]["identical"] for row in rows)]
    return {"steps": rows, "first_difference": first_difference(rows),
            "identical": first_difference(rows) is None,
            "fields_identical_at_every_step": identical}


def git_head(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", type=Path, required=True,
                        help="the hex tree/ directory under test (its src/ is imported)")
    parser.add_argument("--package", default="hexcore")
    parser.add_argument("--init", type=Path, required=True,
                        help="the init file the native run read")
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=1800.0)
    parser.add_argument("--order", type=int, choices=(2, 3), required=True)
    parser.add_argument("--substeps", type=int, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "both"), default="both")
    parser.add_argument("--steps", type=int, default=None, help="native steps to cover")
    parser.add_argument("--cpu-steps", type=int, default=None,
                        help="cap on NumPy steps (each takes minutes)")
    parser.add_argument("--hex-monotonic", choices=("on", "off"), default="on",
                        help="the scalar limiter on the hex side.  Trees before 0.3.4 "
                             "refuse it on the card; with the tracer identically zero, "
                             "as here, it cannot reach a dry field")
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)

    for import_root in (args.tree / "src", args.tree / "tools"):
        sys.path.insert(0, str(import_root))
    pkg = args.package
    driver_module = importlib.import_module(f"{pkg}.driver")
    config_module = importlib.import_module(f"{pkg}.config_v841")
    dynamics_module = importlib.import_module(f"{pkg}.dynamics_v841")
    mesh_module = importlib.import_module(f"{pkg}.mesh")

    config = config_module.V841DryDycoreConfig(
        config_dt=float(args.dt),
        config_time_integration_order=int(args.order),
        config_number_of_sub_steps=int(args.substeps),
        config_split_dynamics_transport=True,
        config_dynamics_split_steps=3,
        config_scalar_advection=True,
        config_monotonic=args.hex_monotonic == "on",
        config_positive_definite=False,
        config_scalar_adv_order=3,
        config_scalar_vadv_order=3,
        config_coef_3rd_order=0.25,
        config_apvm_upwinding=0.5,
        config_smdiv=0.0,
        config_divergence_damping=False,
        config_xnutr=0.0,
        config_zd=22_000.0,
        config_horiz_mixing="off",
    )
    config.validate()
    native_files = sorted(args.native_dir.glob("tracer.*.nc"))
    if len(native_files) < 2:
        raise SystemExit(f"{args.native_dir} holds fewer than two native outputs")
    if args.steps is not None:
        native_files = native_files[: args.steps + 1]
    mesh = mesh_module.Mesh.from_netcdf(args.init)
    native = driver_module.load_mpas_vertical_grid(
        args.init, mesh, config_coef_3rd_order=config.config_coef_3rd_order)
    profiles = dynamics_module.load_v841_reference_wind_profiles(
        args.init, n_vert_levels=native.vertical_grid.n_vert_levels)

    def load(path: Path) -> tuple[Any, Any, Any]:
        return driver_module.load_mpas_initial_state(
            path, mesh, native.vertical_grid, scalar_names=("qv",),
            reference_path=path, terrain_metrics=native.terrain_metrics,
            return_saved_diagnostics=True)

    loaded = [load(path) for path in native_files]
    report: dict[str, Any] = {
        "schema": "gpuwm-hex.native-dry-identity/v1",
        "hex_commit": git_head(args.tree),
        "package": pkg,
        "host": platform.node(),
        "numpy": np.__version__,
        "native_dir": str(args.native_dir),
        "native_outputs": {path.name: sha256_file(path) for path in native_files},
        "init_sha256": sha256_file(args.init),
        "config": {"dt": float(args.dt), "order": int(args.order),
                   "acoustic_substeps": int(args.substeps), "dynamics_splits": 3,
                   "mixing": "off", "physics": "none", "native_monotonic": True,
                   "hex_monotonic": args.hex_monotonic == "on"},
        "cells": int(np.asarray(loaded[0][0].rho).shape[1]),
        "levels": int(np.asarray(loaded[0][0].rho).shape[0]),
        "native_array_of_field": NATIVE_NAME,
    }

    cpu_first = None
    if args.backend in ("cpu", "both"):
        rows = []
        count = len(loaded) - 1
        if args.cpu_steps is not None:
            count = min(count, args.cpu_steps)
        for index in range(count):
            state, reference, saved = loaded[index]
            driver = driver_module.DryDycoreDriver(
                mesh, native.vertical_grid, reference, config,
                terrain_metrics=native.terrain_metrics,
                reference_wind_profiles=profiles)
            result = driver.step(state, saved_diagnostics=saved)
            if index == 0:
                cpu_first = result
            rows.append({"step": index + 1,
                         "fields": compare(result.state, result.saved_diagnostics,
                                           loaded[index + 1][0], loaded[index + 1][2])})
            print(f"cpu step {index + 1}: first difference "
                  f"{first_difference(rows[-1:])}", flush=True)
        report["cpu_one_step_from_each_native_output"] = summary(rows)

    if args.backend in ("cuda", "both"):
        import cupy as cp

        backend = importlib.import_module(f"{pkg}.cuda_backend")
        runtime = importlib.import_module(f"{pkg}.cuda_backend.runtime")
        cuda_driver = importlib.import_module(f"{pkg}.cuda_driver")
        if args.cache_root is None:
            raise SystemExit("--cache-root is required for the cuda backend")
        state, reference, saved = loaded[0]
        arrays = mesh.arrays
        for name in ("defc_a", "defc_b"):
            present = arrays.get(name)
            if present is None or np.asarray(present).dtype.kind != "f":
                counts = np.asarray(arrays["nEdgesOnCell"])
                edges = np.asarray(arrays["edgesOnCell"])
                width = edges.shape[1] if edges.shape[0] == counts.size else edges.shape[0]
                arrays[name] = np.zeros((counts.size, width), dtype=np.float32)
        capability = backend.require_cuda(min_compute=(7, 5))
        cache = runtime.KernelCache(capability=capability, cache_dir=str(args.cache_root))
        driver = cuda_driver.CudaDryDycoreDriver.from_host(
            mesh, state, native.vertical_grid, reference, config,
            saved_diagnostics=saved, terrain_metrics=native.terrain_metrics,
            kernel_cache=cache, reference_wind_profiles=profiles)
        report["card"] = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode()
        report["cupy"] = cp.__version__

        class HostSaved:
            pass

        rows = []
        for index in range(1, len(loaded)):
            result = driver.step_device()
            driver.atmosphere = result.atmosphere
            host = driver.atmosphere.state.to_host()
            host_saved = HostSaved()
            for name in SIDECAR_FIELDS:
                setattr(host_saved, name, cp.asnumpy(getattr(driver.atmosphere.saved, name)))
            rows.append({"step": index,
                         "fields": compare(host, host_saved, loaded[index][0], loaded[index][2])})
            if index == 1 and cpu_first is not None:
                against = compare(host, host_saved, cpu_first.state, cpu_first.saved_diagnostics)
                report["cpu_against_cuda_step_one"] = {
                    "fields": against,
                    "identical": all(entry["identical"] for entry in against.values())}
        report["cuda_continuous_from_native_time_zero"] = summary(rows)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=1), encoding="utf-8")
    brief = {key: ({"identical": value["identical"],
                    "first_difference": value.get("first_difference"),
                    "fields_identical_at_every_step": value.get("fields_identical_at_every_step")}
                   if isinstance(value, dict) and "identical" in value else None)
             for key, value in report.items()}
    print(json.dumps({key: value for key, value in brief.items() if value}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
