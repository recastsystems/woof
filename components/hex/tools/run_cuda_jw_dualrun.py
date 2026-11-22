#!/usr/bin/env python3
"""Run the 24-step x1.2562 CUDA trajectory twice and compare every leaf."""

from __future__ import annotations
import os as _hex_os
_hex_os.environ["WOOF_HEX_NUMERICS"] = "strict"  # a bitwise proof against native MPAS runs strict numerics (fast is the default, strict the opt-in)

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from woof.hex.cuda_backend import require_cuda  # noqa: E402
from woof.hex.cuda_dualrun import (  # noqa: E402
    compare_cuda_capsule_files,
    derive_step_count,
    jw_day_config,
    load_ftz_binding_record,
    load_gpuwm_dualrun,
    prepare_cuda_kernel_cache,
    prepare_jw_inputs,
    run_cuda_arm,
    write_json_atomic,
)


WORK_ROOT = ROOT.parents[1] / "work" / "jw_step"
DEFAULT_INITIAL = WORK_ROOT / "authority_init.nc"
DEFAULT_NATIVE_T0 = WORK_ROOT / "nomix_internal_t0.nc"
DEFAULT_GPUWM_ROOT = Path(os.environ.get("WOOF_ROOT", str(Path.home() / "woof")))
DEFAULT_GPUWM_PROBE = ROOT / "receipts" / "cuda-ftz-sm120" / "gpuwm-probe"
DEFAULT_FTZ_BINDING = ROOT / "receipts" / "cuda-ftz-sm120" / "binding.json"
DEFAULT_OUTPUT = ROOT / "receipts" / "cuda-dualrun-sm120"
DEFAULT_CACHE_ROOT = ROOT / "work" / "cuda-dualrun-sm120-fresh"

EXPECTED_INITIAL_SHA256 = (
    "45c6879f794af984de791ca7da654a7da5d515dbdb6a131ea778f4edcf597970"
)
EXPECTED_NATIVE_T0_SHA256 = (
    "01adfd13c1abe481316a610c875df961938b76b2a12a155ae66e56e348584249"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the native JW state once, upload and run it twice with "
            "independent CUDA caches, hash every state+sidecar step, then use "
            "woof's total capsule comparator."
        )
    )
    parser.add_argument("--initial", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--native-t0", type=Path, default=DEFAULT_NATIVE_T0)
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
    config = jw_day_config(args.dt, args.acoustic_substeps)
    gpuwm_root = args.gpuwm_root.expanduser().resolve(strict=True)
    _, comparison_authority = load_gpuwm_dualrun(gpuwm_root)

    print("validating FTZ/compiler authority", flush=True)
    ftz_binding = load_ftz_binding_record(
        args.ftz_binding,
        gpuwm_root=gpuwm_root,
        gpuwm_receipt_root=args.gpuwm_probe,
    )
    print("preparing frozen JW inputs once", flush=True)
    prepared = prepare_jw_inputs(
        args.initial,
        args.native_t0,
        config,
        expected_initial_sha256=EXPECTED_INITIAL_SHA256,
        expected_native_t0_sha256=EXPECTED_NATIVE_T0_SHA256,
    )

    cache_root = args.cache_root.expanduser().resolve()
    capability = require_cuda(
        min_compute=(12, 0),
        cache_dir=cache_root,
    )
    print("compiling one executable for both CUDA arms", flush=True)
    kernel_cache = prepare_cuda_kernel_cache(
        capability,
        cache_root,
        ftz_binding=ftz_binding,
    )
    output = args.output_root.expanduser().resolve()
    capsule_a_path = output / "JW-x1.2562-24h-dt3600-arm-a.json"
    capsule_b_path = output / "JW-x1.2562-24h-dt3600-arm-b.json"
    report_path = output / "JW-x1.2562-24h-dt3600-comparison.json"

    print(f"running CUDA arm A ({steps} full steps)", flush=True)
    capsule_a = run_cuda_arm(
        prepared,
        config,
        steps=steps,
        kernel_cache=kernel_cache,
        ftz_binding=ftz_binding,
        comparison_authority=comparison_authority,
    )
    write_json_atomic(capsule_a_path, capsule_a)

    print(f"running CUDA arm B ({steps} full steps)", flush=True)
    capsule_b = run_cuda_arm(
        prepared,
        config,
        steps=steps,
        kernel_cache=kernel_cache,
        ftz_binding=ftz_binding,
        comparison_authority=comparison_authority,
    )
    write_json_atomic(capsule_b_path, capsule_b)

    report = compare_cuda_capsule_files(
        capsule_a_path,
        capsule_b_path,
        gpuwm_root=gpuwm_root,
        report_path=report_path,
    )
    summary = {
        "schema": report["schema"],
        "target_steps": steps,
        "target_duration_seconds": float(args.duration),
        "identical": report["gpuwm_comparison"]["identical"],
        "divergence_count": report["gpuwm_comparison"]["divergence_count"],
        "first_divergent_field": report["gpuwm_comparison"]["first_divergent_field"],
        "capsule_sha256": report["capsules"],
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
