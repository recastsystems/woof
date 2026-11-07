#!/usr/bin/env python
"""Run the frozen v8.4.1 full-physics forecast at a REGISTERED mesh shape.

This is the door for every mesh in ``src/hexcore/drivers/mpas_mesh_binding.py``, including
the native ``x4.163842`` -- where the binding is a proven no-op, so the same
command runs the frozen shape without changing it.

    python tools/run_cuda_v841_forecast_mesh.py \
        --repo <port tree> --mesh x1.40962 \
        --receipt-json <path> -- \
        --grid ... --static ... --init ... --hours 0.7 ...

Everything after ``--`` is handed verbatim to ``run_cuda_v841_forecast.main``.
The port tree is never modified: the mesh shape is bound at runtime and
The executing modules are hashed before the bind and again around the run
and recorded with whether they match the x4 proof's pins
(``measure_execution_sources``); since 2026-10-10 a difference is recorded,
not refused.

``--verify-only`` performs the whole bind, including the fail-closed
cross-examination of the mesh files, and stops before the forecast.
``--selftest`` validates the instrument in BOTH directions: a correct bind
must pass, and a deliberately mismatched one must be refused BY NAME.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _extract(rest: list[str], flag: str) -> str | None:
    for index, token in enumerate(rest):
        if token == flag and index + 1 < len(rest):
            return rest[index + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True,
                        help="port tree containing src/ and tools/")
    parser.add_argument("--mesh", required=True,
                        help="registered mesh name, e.g. x1.40962 or x4.163842")
    parser.add_argument("--receipt-json", type=Path, default=None)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--dt", default=None,
                        help="run timestep: 'auto' (default, MPAS-A's 6 s per km of "
                             "nominal spacing), 'row' (the registered value) or seconds")
    parser.add_argument("--max-wind", type=float, default=None,
                        help="largest horizontal wind (m/s) the bind's advective "
                             "stability estimate assumes (default: the mesh row's, 125)")
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    rest = list(args.rest)
    if rest and rest[0] == "--":
        rest = rest[1:]

    repo = args.repo.resolve(strict=True)
    sys.path.insert(0, str(repo / "src"))
    binding_mod = _load(
        "mpas_mesh_binding", repo / "src" / "hexcore" / "drivers" / "mpas_mesh_binding.py"
    )

    grid = _extract(rest, "--grid")
    static = _extract(rest, "--static")
    if grid is None or static is None:
        raise SystemExit(
            "refusing: --grid and --static must both be present after '--' so "
            "the named mesh can be cross-examined against the files"
        )

    forecast = _load(
        "v841_forecast", repo / "src" / "hexcore" / "drivers" / "run_cuda_v841_forecast.py"
    )
    proof = forecast.proof

    if args.selftest:
        return _selftest(binding_mod, proof, forecast, args.mesh, grid, static)

    # The bind measures stability and cadence integrality against what the
    # forecast will execute, so it takes the same schedule the forecast's own
    # arguments name (MPAS-A defaults when they name none).
    from woof.hex import mpas_defaults

    row = binding_mod.MESH_BINDINGS.get(args.mesh)
    radiation = float(_extract(rest, "--radiation-seconds")
                      or mpas_defaults.DEFAULT_RADIATION_SECONDS)
    dt_choice = mpas_defaults.resolve_dt(
        args.dt,
        nominal_dx_m=None if row is None else float(row.nominal_dx_m),
        row_dt_seconds=None if row is None else float(row.dt_seconds),
        radiation_seconds=radiation,
    )
    receipt = binding_mod.bind_mesh(
        proof, args.mesh, grid=Path(grid), static=Path(static),
        forecast=forecast,
        dt_seconds=float(dt_choice["dt_seconds"]),
        time_integration_order=int(_extract(rest, "--time-integration-order")
                                   or mpas_defaults.DEFAULT_TIME_INTEGRATION_ORDER),
        acoustic_substeps=int(_extract(rest, "--acoustic-substeps")
                              or mpas_defaults.DEFAULT_ACOUSTIC_SUBSTEPS),
        radiation_seconds=radiation,
        convection=_extract(rest, "--convection") or "auto",
        max_wind_m_s=args.max_wind,
    )
    receipt["repo"] = str(repo)
    receipt["argv_after_separator"] = rest

    if args.verify_only:
        print("[mesh-run] --verify-only: bind complete, forecast NOT run")
        receipt["rc"] = 0
        receipt["forecast_run"] = False
        if args.receipt_json:
            args.receipt_json.write_text(json.dumps(receipt, indent=1))
        return 0

    started = time.perf_counter()
    rc = 1
    try:
        rc = forecast.main(rest)
    finally:
        receipt["rc"] = rc
        receipt["forecast_run"] = True
        receipt["wall_seconds"] = time.perf_counter() - started
        if args.receipt_json:
            args.receipt_json.write_text(json.dumps(receipt, indent=1))
        print(
            f"[mesh-run] mesh={args.mesh} rc={rc} "
            f"wall={receipt['wall_seconds']:.3f}s"
        )
    return rc


def _selftest(binding_mod, proof, forecast, mesh: str, grid: str, static: str) -> int:
    """Validate the instrument in both directions, and print its limits."""
    registry = binding_mod.MESH_BINDINGS
    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"[selftest] {'PASS' if ok else 'FAIL'}  {label}  {detail}")
        if not ok:
            failures.append(label)

    # Positive direction: the correct mesh binds.  The baseline comes from
    # binding_fingerprint so it is the same call, with the same edge-length
    # authority, that bind_mesh compares against -- a bare constants_fingerprint
    # digests a null authority and could never match a no-op bind.
    fingerprint0 = binding_mod.binding_fingerprint(proof, Path(static))["sha256"]
    # The frozen native row is a no-op only at the schedule the native
    # reference was recorded at, which is passed explicitly (since
    # 2026-10-10 the defaults are MPAS-A's, not x4's).
    row = registry.get(mesh)
    recorded = {}
    if row is not None:
        recorded["dt_seconds"] = float(row.dt_seconds)
        if row.frozen_native:
            recorded.update(time_integration_order=3, acoustic_substeps=6,
                            radiation_seconds=600.0)
    receipt = binding_mod.bind_mesh(
        proof, mesh, grid=Path(grid), static=Path(static), forecast=forecast,
        log=lambda *a, **k: None, **recorded,
    )
    check(
        f"correct bind accepted ({mesh})",
        receipt["mesh"] == mesh,
        f"fingerprint {fingerprint0[:12]} -> {receipt['constants_fingerprint_after'][:12]}",
    )
    check(
        "treatment engaged (fingerprint moved) or native no-op (unchanged)",
        (receipt["rebound"] and
         receipt["constants_fingerprint_after"] != fingerprint0)
        or (not receipt["rebound"] and
            receipt["constants_fingerprint_after"] == fingerprint0),
    )

    # Negative direction 1: name a different registered mesh, same files.
    other = next((n for n in registry if n != mesh), None)
    if other is not None:
        try:
            binding_mod.bind_mesh(
                proof, other, grid=Path(grid), static=Path(static),
                forecast=None, verify_frozen_sources=False,
                log=lambda *a, **k: None,
            )
            check(f"wrong mesh name refused ({other} vs {mesh} files)", False,
                  "IT WAS ACCEPTED")
        except binding_mod.MeshBindingMismatch as exc:
            check(f"wrong mesh name refused ({other} vs {mesh} files)", True,
                  f"{type(exc).__name__}: {str(exc)[:90]}")

    # Negative direction 2: an unregistered mesh.
    try:
        binding_mod.bind_mesh(
            proof, "x9.999999", grid=Path(grid), static=Path(static),
            forecast=None, verify_frozen_sources=False,
            log=lambda *a, **k: None,
        )
        check("unregistered mesh refused", False, "IT WAS ACCEPTED")
    except binding_mod.MeshBindingMismatch as exc:
        check("unregistered mesh refused", True,
              f"{type(exc).__name__}: {str(exc)[:90]}")

    # Negative direction 3: a real file that is not the declared one.
    try:
        binding_mod.bind_mesh(
            proof, mesh, grid=Path(static), static=Path(static),
            forecast=None, verify_frozen_sources=False,
            log=lambda *a, **k: None,
        )
        check("swapped grid/static file refused", False, "IT WAS ACCEPTED")
    except binding_mod.MeshBindingMismatch as exc:
        check("swapped grid/static file refused", True,
              f"{type(exc).__name__}: {str(exc)[:90]}")

    print(
        "[selftest] RESOLUTION LIMIT: this instrument proves the mesh SHAPE "
        "and the mesh FILE IDENTITY agree with the name, and that the six "
        "frozen sources are unchanged.  It proves NOTHING about trajectory "
        "correctness at the bound width -- work grouped by width still needs "
        "its own bit-identity gate, on one card in one session."
    )
    print(f"[selftest] {'ALL PASS' if not failures else 'FAILURES: ' + repr(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
