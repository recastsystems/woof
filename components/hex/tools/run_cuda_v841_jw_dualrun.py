#!/usr/bin/env python3
"""Run the held v8.4.1 JW CUDA durability profile twice and compare all leaves.

This is deterministic implementation evidence, not a native-weather authority
claim.  It deliberately requires scalar advection with one nonzero tracer,
positive APVM upwinding, and the exact non-FCT lane.  Both arms share one
prepared executable but independently upload the same sealed host preparation.
"""

from __future__ import annotations
import os as _hex_os
_hex_os.environ["WOOF_HEX_NUMERICS"] = "strict"  # a bitwise proof against native MPAS runs strict numerics (fast is the default, strict the opt-in)

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from woof.hex.config_v841 import V841DryDycoreConfig  # noqa: E402
from woof.hex.cuda_backend import require_cuda  # noqa: E402
from woof.hex.cuda_dualrun import (  # noqa: E402
    PreparedCudaInputs,
    compare_cuda_capsule_files,
    derive_step_count,
    load_ftz_binding_record,
    load_gpuwm_dualrun,
    prepare_cuda_kernel_cache,
    run_cuda_arm_generic,
    sha256_file,
    write_json_atomic,
)
from woof.hex.driver import (  # noqa: E402
    load_mpas_initial_state,
    load_mpas_vertical_grid,
)
from woof.hex.dynamics_v841 import (  # noqa: E402
    load_v841_reference_wind_profiles,
)
from woof.hex.mesh import Mesh  # noqa: E402


DEFAULT_FIXTURE = ROOT / "oracle" / "jw-x1.2562-v8.4.1-split3-endpoint-nonclaim"
DEFAULT_GPUWM_ROOT = Path(os.environ.get("WOOF_ROOT", str(Path.home() / "woof")))
DEFAULT_GPUWM_PROBE = ROOT / "receipts" / "cuda-ftz-sm120" / "gpuwm-probe"
DEFAULT_FTZ_BINDING = ROOT / "receipts" / "cuda-ftz-sm120-v841" / "binding.json"
DEFAULT_OUTPUT = ROOT / "receipts" / "cuda-dualrun-sm120-v841"
DEFAULT_CACHE_ROOT = ROOT / "work" / "cuda-dualrun-sm120-v841-fresh"

FIXTURE_PINS = {
    "manifest.json": "500a33646c4a164564e5dc0d50f833eda3dc959a1abe91ae991fe59ebb6b23e6",
    "reference-input.nc": (
        "45c6879f794af984de791ca7da654a7da5d515dbdb6a131ea778f4edcf597970"
    ),
    "endpoint-t0.nc": (
        "53daf8bb17f28724db41d88c1d007d1d2f845a9f5e340662ddbd01d68f7f60b3"
    ),
}


def _artifact_record(path: Path, expected_sha256: str) -> dict[str, object]:
    selected = path.resolve(strict=True)
    digest = sha256_file(selected)
    if digest != expected_sha256:
        raise RuntimeError(
            f"fixture input changed at {selected.name}: {digest} != {expected_sha256}"
        )
    return {"bytes": selected.stat().st_size, "sha256": digest}


def _nonzero_tracer(state: object, mesh: Mesh) -> np.ndarray:
    source = np.asarray(getattr(state, "scalars"))
    if source.ndim != 3 or source.shape[0] < 1:
        raise RuntimeError("v8.4.1 JW preparation has no transported scalar slot")
    ntracers, nlev, ncells = source.shape
    lon = np.asarray(mesh.lonCell, dtype=np.float32)
    lat = np.asarray(mesh.latCell, dtype=np.float32)
    if lon.shape != (ncells,) or lat.shape != (ncells,):
        raise RuntimeError("JW mesh coordinate shape differs from scalar cells")
    horizontal = np.float32(0.001) + np.float32(0.0002) * (
        np.float32(0.5)
        + np.float32(0.5)
        * np.sin(lon, dtype=np.float32)
        * np.cos(lat, dtype=np.float32)
    )
    level = np.arange(nlev, dtype=np.float32)
    vertical = np.float32(1.0) - np.float32(0.2) * level / np.float32(max(nlev - 1, 1))
    tracer = np.empty((ntracers, nlev, ncells), dtype=np.float32)
    tracer[0] = vertical[:, None] * horizontal[None, :]
    for index in range(1, ntracers):
        tracer[index] = tracer[0] * np.float32(1.0 / (index + 1))
    result = np.ascontiguousarray(tracer)
    if not np.all(np.isfinite(result)) or not np.any(result != np.float32(0.0)):
        raise RuntimeError("deterministic transported tracer is vacuous")
    return result


def _config(dt: float, acoustic_substeps: int) -> V841DryDycoreConfig:
    return V841DryDycoreConfig(
        config_dt=dt,
        config_time_integration_order=3,
        config_number_of_sub_steps=acoustic_substeps,
        config_split_dynamics_transport=True,
        config_dynamics_split_steps=3,
        config_scalar_advection=True,
        config_monotonic=False,
        config_positive_definite=False,
        config_scalar_adv_order=3,
        config_scalar_vadv_order=3,
        config_coef_3rd_order=0.25,
        config_apvm_upwinding=0.5,
        config_horiz_mixing="off",
        config_smdiv=0.0,
        config_xnutr=0.0,
        config_zd=22_000.0,
    )


def _prepare(fixture: Path, config: V841DryDycoreConfig) -> PreparedCudaInputs:
    selected = fixture.expanduser().resolve(strict=True)
    files = {name: selected / name for name in FIXTURE_PINS}
    input_bytes = {
        name: _artifact_record(files[name], expected)
        for name, expected in FIXTURE_PINS.items()
    }
    reference_input = files["reference-input.nc"]
    t0 = files["endpoint-t0.nc"]
    mesh = Mesh.from_netcdf(reference_input)
    native = load_mpas_vertical_grid(
        reference_input,
        mesh,
        config_coef_3rd_order=config.config_coef_3rd_order,
    )
    state, reference, saved = load_mpas_initial_state(
        t0,
        mesh,
        native.vertical_grid,
        scalar_names=("qv",),
        reference_path=t0,
        terrain_metrics=native.terrain_metrics,
        return_saved_diagnostics=True,
    )
    tracer = _nonzero_tracer(state, mesh)
    state = replace(state, scalars=tracer)
    input_bytes["transported-tracer-t0"] = {
        "bytes": int(tracer.nbytes),
        "sha256": hashlib.sha256(tracer.tobytes(order="C")).hexdigest(),
    }
    profiles = load_v841_reference_wind_profiles(
        reference_input,
        n_vert_levels=native.vertical_grid.n_vert_levels,
    )
    return PreparedCudaInputs.validated(
        config=config,
        profile="jw-x1.2562-v841-split3-nonzero-tracer-durability",
        target="24-step JW x1.2562 v8.4.1 CUDA deterministic dual run",
        preparation_method=(
            "source-pinned compiled-endpoint t0 with one deterministic nonzero "
            "transported tracer; one sealed host preparation, two independent uploads"
        ),
        mesh=mesh,
        state=state,
        vertical=native.vertical_grid,
        reference=reference,
        saved_diagnostics=saved,
        terrain_metrics=native.terrain_metrics,
        input_bytes=input_bytes,
        reference_wind_profiles=profiles,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the held v8.4.1 24-step JW profile twice with a shared compiled "
            "executable and independent uploads, then compare every capsule leaf."
        )
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--gpuwm-root", type=Path, default=DEFAULT_GPUWM_ROOT)
    parser.add_argument("--gpuwm-probe", type=Path, default=DEFAULT_GPUWM_PROBE)
    parser.add_argument("--ftz-binding", type=Path, default=DEFAULT_FTZ_BINDING)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--dt", type=float, default=3600.0)
    parser.add_argument("--duration", type=float, default=86_400.0)
    parser.add_argument("--acoustic-substeps", type=int, default=6)
    return parser


def execute(args: argparse.Namespace) -> int:
    steps = derive_step_count(args.duration, args.dt)
    if (
        steps != 24
        or float(args.dt) != 3600.0
        or float(args.duration) != 86_400.0
        or int(args.acoustic_substeps) != 6
    ):
        raise RuntimeError(
            "v8.4.1 JW certification profile requires exactly 24 dt=3600 steps "
            "with six acoustic substeps"
        )
    config = _config(args.dt, args.acoustic_substeps)
    config.validate()
    gpuwm_root = args.gpuwm_root.expanduser().resolve(strict=True)
    _, comparison_authority = load_gpuwm_dualrun(gpuwm_root)

    print("validating live v8.4.1 FTZ/compiler authority", flush=True)
    ftz_binding = load_ftz_binding_record(
        args.ftz_binding,
        gpuwm_root=gpuwm_root,
        gpuwm_receipt_root=args.gpuwm_probe,
        source_release="v8.4.1",
    )
    print("sealing one v8.4.1 JW nonzero-tracer preparation", flush=True)
    prepared = _prepare(args.fixture, config)

    cache_root = args.cache_root.expanduser().resolve()
    capability = require_cuda(
        min_compute=(12, 0),
        cache_dir=cache_root,
    )
    print("compiling one exact eight-TU executable for both arms", flush=True)
    kernel_cache = prepare_cuda_kernel_cache(
        capability,
        cache_root,
        ftz_binding=ftz_binding,
        source_release="v8.4.1",
    )

    output = args.output_root.expanduser().resolve()
    paths = {
        "a": output / "JW-x1.2562-v841-24h-dt3600-arm-a.json",
        "b": output / "JW-x1.2562-v841-24h-dt3600-arm-b.json",
        "comparison": output / "JW-x1.2562-v841-24h-dt3600-comparison.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise RuntimeError(f"refusing to overwrite existing evidence: {existing}")

    print(f"running CUDA arm A ({steps} resident steps)", flush=True)
    arm_a = run_cuda_arm_generic(
        prepared,
        config,
        steps=steps,
        kernel_cache=kernel_cache,
        ftz_binding=ftz_binding,
        comparison_authority=comparison_authority,
    )
    write_json_atomic(paths["a"], arm_a.capsule)

    print(f"running CUDA arm B ({steps} independent-upload resident steps)", flush=True)
    arm_b = run_cuda_arm_generic(
        prepared,
        config,
        steps=steps,
        kernel_cache=kernel_cache,
        ftz_binding=ftz_binding,
        comparison_authority=comparison_authority,
    )
    write_json_atomic(paths["b"], arm_b.capsule)

    report = compare_cuda_capsule_files(
        paths["a"],
        paths["b"],
        gpuwm_root=gpuwm_root,
        report_path=paths["comparison"],
    )
    summary = {
        "schema": report["schema"],
        "source_release": "v8.4.1",
        "steps": steps,
        "duration_seconds": float(args.duration),
        "configuration": {
            "config_scalar_advection": config.config_scalar_advection,
            "config_monotonic": config.config_monotonic,
            "config_positive_definite": config.config_positive_definite,
            "config_apvm_upwinding": config.config_apvm_upwinding,
        },
        "transported_tracers": int(prepared.state.scalars.shape[0]),
        "internal_d2h_bytes_per_step": 4,
        "identical": report["gpuwm_comparison"]["identical"],
        "divergence_count": report["gpuwm_comparison"]["divergence_count"],
        "first_divergent_field": report["gpuwm_comparison"]["first_divergent_field"],
        "capsules": report["capsules"],
        "comparison_authority_sha256": comparison_authority["source_sha256"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if report["gpuwm_comparison"]["identical"] is True else 2


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return execute(args)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
