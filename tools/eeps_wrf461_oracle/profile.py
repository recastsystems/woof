#!/usr/bin/env python3
"""Profile intact EEPS phases without changing the production kernel.

The default input is an archived synthetic native-module trace. This is a
cost diagnostic, separate from real-input replay and whole-forecast proof.
Clock64 intervals include scheduling and cache effects for each column;
their fractions are not an additive attribution of whole-device wall time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cupy as cp
import numpy as np

from woof.core import eeps_pbl
from woof.core.health_ledger import HealthLedger, deferring
from woof.verify.eeps_oracle import read_trace


def instrument(source):
    """Insert only clock reads, compiler fences and independent stores."""
    def replace(before, after):
        nonlocal source
        assert source.count(before) == 1, before
        source = source.replace(before, after)

    replace("workspace, const int column_offset)",
            "workspace, const int column_offset, unsigned long long* profile)")
    replace("    const int km = EEPS_KM;", """    const int km = EEPS_KM;
    asm volatile("" ::: "memory");
    unsigned long long profile_start = clock64();
#define EEPS_STAMP(phase) do { \\
    asm volatile("" ::: "memory"); \\
    unsigned long long now = clock64(); \\
    profile[(size_t)(phase) * n + col] = now - profile_start; \\
    profile_start = now; \\
    asm volatile("" ::: "memory"); \\
} while (0)
""")
    for marker, phase in (
        ("    // :539-562 first-guess PBLH", "0"),
        ("    // :796-898 time integration", "1"),
    ):
        replace(marker, f"    EEPS_STAMP({phase});\n" + marker)
    replace("        // TKE (:872-895)",
            "        EEPS_STAMP(2 + 2 * (lvl - 1));\n\n        // TKE (:872-895)")
    marker = ("                pek[k] = eeps_min(EEPS_MAXPEK, eeps_max(EEPS_MINPEK, d));\n"
              "            }\n        }\n")
    replace(marker, marker + "        EEPS_STAMP(3 + 2 * (lvl - 1));\n")
    replace("    if (flags) atomicOr(status, flags);",
            "    if (flags) atomicOr(status, flags);\n"
            "    EEPS_STAMP(2 + 2 * mdt);\n#undef EEPS_STAMP")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=Path(__file__).resolve().parents[2]
                        / "tests/data/oracles/eeps/trace-A.bin.gz")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=12.)
    args = parser.parse_args()
    from woof import numerics
    if numerics.mode() != "fast":
        parser.error("this cost diagnostic requires GPUWM_NUMERICS=fast")
    trace = read_trace(args.trace)
    inputs = trace["steps"][0]["inputs"]
    nz, ny, nx = trace["header"]["nz"], 150, 150
    n = ny * nx
    free, total = cp.cuda.runtime.memGetInfo()
    if free < 1024 ** 3:
        raise RuntimeError(f"wait for 1 GiB free card memory; currently {free} bytes")

    def expand(array):
        array = np.asarray(array)
        repeated = np.tile(array, (1, (n + array.shape[-1] - 1) // array.shape[-1])
                           if array.ndim == 2 else (n + array.size - 1) // array.size)
        return cp.ascontiguousarray(cp.asarray(repeated[..., :n]).reshape(
            (nz, ny, nx) if array.ndim == 2 else (ny, nx)))

    columns = {key: expand(inputs[key]) for key in eeps_pbl.COLUMN_INPUTS}
    surface = {key: expand(inputs[key]) for key in eeps_pbl.SURFACE_INPUTS}
    seed = {key: expand(inputs[source]) for key, source in
            zip(eeps_pbl.CARRIED, ("pek", "pep", "pek_adv", "pep_adv"))}
    carried = {key: array.copy() for key, array in seed.items()}
    mdt = 1 + int(np.sqrt(np.float32(args.dt / 9.)) + np.float32(.00001))
    names = ["column_packing_thermodynamics_and_richardson",
             "pblh_surface_and_topdown_source"]
    for substep in range(mdt):
        names += [f"substep_{substep + 1}_eddy_sources_and_epsilon_solve",
                  f"substep_{substep + 1}_tke"]
    # The six atmospheric solves are fused with the output conversion; a
    # fence between their sweeps moves fast-mode contraction, so they are
    # one interval.
    names += ["diffusivities_six_atmospheric_solves_output_and_health"]
    stamps = cp.zeros((len(names), n), dtype=cp.uint64)
    source = eeps_pbl.module_source(nz)
    profiled = instrument(source)
    # Match the production fast RawModule route, including CuPy's FTZ
    # setting and the shared numerics compiler hook.
    from woof.core.kernels import module_options
    module = cp.RawModule(code=profiled, options=module_options("eeps"),
                          name_expressions=None)
    function = module.get_function("eeps_column")

    class WithClock:
        kernel = getattr(function, "kernel", function)

        def __call__(self, grid, block, arguments):
            function(grid, block, arguments + (stamps,))

    ledger = HealthLedger(label="eeps-phase-profile")

    def call(kernel=None):
        with deferring(ledger):
            return eeps_pbl.eeps_step(columns, surface, carried, dt=args.dt,
                                      first_step=True, kernel=kernel)

    def restore():
        for key, array in carried.items():
            cp.copyto(array, seed[key])

    restore()
    baseline = call()
    baseline.update({key: array.copy() for key, array in carried.items()})
    restore()
    timed = call(WithClock())
    timed.update(carried)
    different = {key: int(cp.count_nonzero(
        array.view(cp.uint32) != timed[key].view(cp.uint32)).item())
                 for key, array in baseline.items()}
    # Fast mode permits FMA contraction.  With the clock fences and the
    # extra argument, the 2026-11-02 fused kernel writes -0.0 where the
    # production build writes +0.0 in 5,000 RVBLTEN words (a zero tendency
    # formed by a contracted subtraction); every value compares equal.
    # Refuse any word whose value differs; record the signed-zero words.
    different_values = {key: int(cp.count_nonzero(array != timed[key]).item())
                        for key, array in baseline.items()}
    signed_zero_words = {key: different[key] - different_values[key]
                         for key in different}
    assert not any(different_values.values()), different_values

    def measure(kernel=None):
        for _ in range(3):
            restore()
            call(kernel)
        cp.cuda.get_current_stream().synchronize()
        samples = []
        for _ in range(15):
            restore()
            start, end = cp.cuda.Event(), cp.cuda.Event()
            start.record()
            call(kernel)
            end.record()
            end.synchronize()
            samples.append(float(cp.cuda.get_elapsed_time(start, end)))
        return {"median_ms": float(np.median(samples)), "samples_ms": samples}

    normal = measure()
    clocked = measure(WithClock())
    cycles = stamps.get()
    ledger.close()
    mean = cycles.mean(axis=1)
    rows = [{"phase": name, "mean_cycles": float(value),
             "median_cycles": float(np.median(cycles[index])),
             "fraction_of_mean_column_cycles": float(value / mean.sum())}
            for index, (name, value) in enumerate(zip(names, mean))]
    result = {
        "schema": "eeps-phase-profile-v1", "input_kind": "synthetic-native-regression",
        "trace": str(args.trace), "trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
        "shape": [nz, ny, nx], "dt": args.dt, "native_substeps": mdt,
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "cupy_version": cp.__version__,
        "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
        "production_kernel_attributes": dict(eeps_pbl._kernel(nz).attributes),
        "instrumented_kernel_attributes": dict(function.attributes),
        "free_card_bytes_before_allocation": free, "total_card_bytes": total,
        "production_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "instrumented_source_sha256": hashlib.sha256(profiled.encode()).hexdigest(),
        "all_output_and_carried_differing_words": different,
        "all_output_and_carried_differing_values": different_values,
        "all_output_and_carried_signed_zero_words": signed_zero_words,
        "all_output_and_carried_compared_words": {
            key: int(array.size) for key, array in baseline.items()},
        "production": normal, "instrumented": clocked,
        "instrumentation_wall_ratio": clocked["median_ms"] / normal["median_ms"],
        "phases": rows,
        "prognostic_phase_fraction": float(mean[2:2 + 2 * mdt].sum() / mean.sum()),
        "prognostic_tridiagonal_solves_per_column": 2 * mdt,
        "prognostic_solved_rows_per_column": 2 * mdt * (nz - 1),
        "atmospheric_tridiagonal_solves_per_column": 6,
        "atmospheric_solved_rows_per_column": 6 * nz,
        "method": "Clock64 fenced intervals in each intact column, all equations retained. "
                  "Fractions include scheduling/cache effects and are not additive device wall attribution. "
                  "Fifteen CUDA-event samples after three warmups measure instrumentation overhead. "
                  "Inputs are archived synthetic columns, distinct from the real forecast stage benchmark.",
    }
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
