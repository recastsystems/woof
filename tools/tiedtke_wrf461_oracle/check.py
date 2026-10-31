"""Compare every cu6 output word against WRF's original Fortran driver.

--host checks the CUDA transcription with CPU intrinsic shims. --gpu runs
the production CuPy loader and requires explicit compute authority. Receipts
contain hashes and differences; no raw column fields are saved.
"""
from __future__ import annotations
import argparse
import ctypes as c
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from woof.core.tiedtke import INPUTS, LEVEL_OUTPUTS, SURFACE_OUTPUTS, workspace_shapes

FIELDS = LEVEL_OUTPUTS + SURFACE_OUTPUTS + ("cu_act_flag",)
REGIMES = ("convective", "shallow", "stable", "tropical-ocean", "high-terrain", "edge")
FP = c.POINTER(c.c_float)
IP = c.POINTER(c.c_int)


def libm_identity():
    """The libc version and the libm actually mapped into this process.

    The Fortran's tan() is that libm's tanf, so a receipt names it: the
    reference is glibc 2.43 (glibc243.sh), whose tanf is correctly rounded.
    """
    libc = c.CDLL("libc.so.6")
    libc.gnu_get_libc_version.restype = c.c_char_p
    c.CDLL("libm.so.6")
    paths = sorted({line.split()[-1] for line in open("/proc/self/maps", encoding="utf-8")
                    if "/libm.so" in line or "/libm-" in line})
    return {"glibc": libc.gnu_get_libc_version().decode(),
            "libm": [{"path": p, "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()}
                     for p in paths]}


def pointer(array):
    return array.ctypes.data_as(IP if array.dtype == np.int32 else FP)


def compare_words(expected, actual):
    """A word comparison plus signed-order ULP distances, including -0."""
    if expected.dtype != actual.dtype or expected.shape != actual.shape:
        raise ValueError("oracle fields have different dtypes or shapes")
    if expected.dtype == np.float32:
        from woof.core.fp32_ulp import monotone_fp32_key

        a, b = expected.view(np.uint32), actual.view(np.uint32)
        # The shared monotone key orders every finite word; -0 and +0 are
        # one point on it (a bit difference still fails through ``bad``).
        # Nonfinite outputs always fail, including identical NaNs or
        # infinities on both sides.
        nonfinite = ~(np.isfinite(expected) & np.isfinite(actual))
        distance = np.abs(monotone_fp32_key(expected) - monotone_fp32_key(actual))
    else:
        a, b = expected, actual
        distance = np.abs(a.astype(np.int64) - b.astype(np.int64))
        nonfinite = np.zeros(a.shape, dtype=bool)
    bad = (a != b) | nonfinite
    finite = ~nonfinite
    # Absolute sizes bound a named residual (an older libm's tangent) in
    # physical units; ULP alone overstates differences near zero.
    delta = np.abs(expected[finite].astype(np.float64) - actual[finite].astype(np.float64))
    return {"differing_words": int(np.count_nonzero(bad)),
            "max_abs_difference": float(delta.max(initial=0.0)),
            "max_abs_expected": float(np.abs(expected[finite].astype(np.float64)).max(initial=0.0)),
            "nonfinite_words": int(np.count_nonzero(nonfinite)),
            "max_ulp": int(np.max(distance, initial=0)),
            "first_difference": np.argwhere(bad)[0].tolist() if np.any(bad) else None,
            "expected_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
            "actual_sha256": hashlib.sha256(actual.tobytes()).hexdigest()}


def planted_check(expected):
    actual = expected.copy()
    location = tuple(np.unravel_index(np.flatnonzero(np.isfinite(expected))[0], expected.shape))
    actual[location] = np.nextafter(expected[location], np.float32(np.inf))
    result = compare_words(expected, actual)
    if result["differing_words"] != 1 or result["max_ulp"] != 1:
        raise AssertionError(f"planted 1 ULP change was not detected: {result}")
    return result


def outputs(nc, nz):
    result = {name: np.full((nz + 1, nc), np.float32(-7.25), dtype=np.float32)
              for name in LEVEL_OUTPUTS}
    result.update({name: np.full(nc, np.float32(-7.25), dtype=np.float32) for name in SURFACE_OUTPUTS})
    result["cu_act_flag"] = np.zeros(nc, dtype=np.int32)
    return result


def fixture(lib, nc, nz, unsafe_top=False):
    result = {name: np.empty((nz + 1, nc), dtype=np.float32) for name in INPUTS}
    for name in ("xland", "qfx"):
        result[name] = np.empty(nc, dtype=np.float32)
    result["znu"] = np.empty(nz + 1, dtype=np.float32)
    regimes = np.empty(nc, dtype=np.int32)
    fn = lib.cu6_fixture
    fn.argtypes = [c.c_int, c.c_int] + [FP] * len(INPUTS) + [IP, c.c_int]
    fn.restype = None
    fn(nc, nz, *(pointer(result[n]) for n in INPUTS), pointer(regimes), int(unsafe_top))
    return result, regimes


def fortran(lib, inputs, *, dt, itimestep, stepcu, fqc, fqi):
    nz, nc = inputs["t3d"].shape
    nz -= 1
    result = outputs(nc, nz)
    fn = lib.cu6_oracle
    fn.argtypes = [c.c_int, c.c_int, c.c_float, c.c_int, c.c_int] + [FP] * (
        len(INPUTS) + len(LEVEL_OUTPUTS) + len(SURFACE_OUTPUTS)) + [IP, c.c_int, c.c_int]
    fn.restype = None
    fn(nc, nz, dt, itimestep, stepcu, *(pointer(inputs[n]) for n in INPUTS),
       *(pointer(result[n]) for n in FIELDS), fqc, fqi)
    return result


def host(lib, inputs, *, dt, itimestep, stepcu, fqc, fqi):
    nz, nc = inputs["t3d"].shape
    nz -= 1
    result = outputs(nc, nz)
    shapes = workspace_shapes(nz, nc)
    workspace = tuple(np.zeros(s, dtype=d) for s, d in zip(
        shapes, (np.float32, np.int32, np.bool_), strict=True))
    fn = lib.tiedtke_column
    fn.argtypes = [FP] * (len(INPUTS) + len(LEVEL_OUTPUTS) + len(SURFACE_OUTPUTS)) + [
        IP, FP, IP, c.c_void_p, c.c_float] + [c.c_int] * 8
    fn.restype = None
    lib.cu6_set_column.argtypes = [c.c_int]
    lib.cu6_set_column.restype = None
    args = [*(pointer(inputs[n]) for n in INPUTS), *(pointer(result[n]) for n in FIELDS),
            pointer(workspace[0]), pointer(workspace[1]), c.c_void_p(workspace[2].ctypes.data),
            dt, itimestep, stepcu, nz, nc, fqc, fqi, 0, 0]
    for column in range(nc):
        lib.cu6_set_column(column)
        fn(*args)
    if nc > 1 and np.any(workspace[1][-1] != 0):
        args[-2] = 1
        for column in range(nc):
            lib.cu6_set_column(column)
            fn(*args)
    return result


def gpu(inputs, *, dt, itimestep, stepcu, fqc, fqi, row_width=None):
    import cupy as cp
    from woof.core.tiedtke import column_step
    nz, nc = inputs["t3d"].shape
    nz -= 1
    device = {name: cp.asarray(a if name in ("w", "p8w", "xland", "qfx")
                              else a[:nz]) for name, a in inputs.items()}
    initial = outputs(nc, nz)
    target = {name: cp.asarray(a[:nz] if name in LEVEL_OUTPUTS else a)
              for name, a in initial.items()}
    answer = column_step(device, dt=dt, itimestep=itimestep, stepcu=stepcu,
                         f_qc=bool(fqc), f_qi=bool(fqi), outputs=target, row_width=row_width)
    cp.cuda.get_current_stream().synchronize()
    result = initial
    for name in FIELDS:
        if name in LEVEL_OUTPUTS:
            result[name][:nz] = cp.asnumpy(answer[name])
        else:
            result[name] = cp.asnumpy(answer[name])
    return result


def rows_reference(lib, inputs, width, **cfg):
    """WRF's own row loop: one original driver call per i row of WIDTH columns."""
    nc = inputs["t3d"].shape[1]
    parts = []
    for lo in range(0, nc, width):
        row = {name: a if name == "znu" else np.ascontiguousarray(
            a[lo:lo + width] if a.ndim == 1 else a[:, lo:lo + width]) for name, a in inputs.items()}
        parts.append(fortran(lib, row, **cfg))
    return {name: np.concatenate([p[name] for p in parts], axis=-1) for name in FIELDS}


#: Row widths the GPU check also launches as several rows in one call. The
#: fixture's six input families are consecutive blocks of 12 columns, so 12
#: makes whole rows with no active column (the gather branch) and 8 makes
#: rows that straddle two families.
ROW_WIDTHS = (12, 8)


def run(args):
    reference = c.CDLL(str(args.build / "libcu6_oracle.so"))
    checked = c.CDLL(str(args.build / "libcu6_checked.so"))
    implementation = c.CDLL(str(args.build / ("libcu6_native_tan_control.so" if args.native_tan_control
                                             else "libcu6_host.so"))) if args.host else None
    all_results = []
    failed = False
    for nz in (4, 12, 50, 72):
        inputs, regimes = fixture(reference, 72, nz, args.unsafe_top)
        input_hashes = {name: hashlib.sha256(a.tobytes()).hexdigest() for name, a in inputs.items()}
        for mode in ((60., 2, 1, 1, 1), (30., 1, 1, 1, 1), (20., 2, 3, 0, 0)):
            cfg = dict(zip(("dt", "itimestep", "stepcu", "fqc", "fqi"), mode, strict=True))
            diagnostic = fortran(checked, inputs, **cfg)
            expect = fortran(reference, inputs, **cfg)
            actual = host(implementation, inputs, **cfg) if args.host else gpu(inputs, **cfg)
            fields = {name: compare_words(expect[name], actual[name]) for name in FIELDS}
            checked_fields = {name: compare_words(expect[name], diagnostic[name]) for name in FIELDS}
            # Batch-to-single checks detect cross-column coupling in WRF.
            isolation = {name: 0 for name in FIELDS}
            for col in range(72):
                single = {name: a if name == "znu" else np.ascontiguousarray(
                    a[col:col+1] if a.ndim == 1 else a[:, col:col+1]) for name, a in inputs.items()}
                isolated = fortran(reference, single, **cfg)
                for name in FIELDS:
                    packed = expect[name][col:col+1] if expect[name].ndim == 1 else expect[name][:, col:col+1]
                    isolation[name] += compare_words(packed, isolated[name])["differing_words"]
            regimes_summary = {}
            for r, label in enumerate(REGIMES, 1):
                mask = regimes == r
                regimes_summary[label] = {"columns": int(np.count_nonzero(mask)),
                    "positive_rain_columns": int(np.count_nonzero(expect["raincv"][mask] > 0)),
                    "max_raincv": float(np.max(expect["raincv"][mask], initial=0))}
            row_batched = {}
            if not args.host:
                for width in ROW_WIDTHS:
                    reference_rows = rows_reference(reference, inputs, width, **cfg)
                    batched = gpu(inputs, **cfg, row_width=width)
                    row_batched[str(width)] = {name: compare_words(reference_rows[name], batched[name])
                                               for name in FIELDS}
                    if any(v["differing_words"] for v in row_batched[str(width)].values()):
                        failed = True
            all_results.append({"nz": nz, "columns": 72, "config": cfg, "fields": fields,
                                "row_batched": row_batched,
                                "checked_oracle": checked_fields, "batch_to_single_differing_words": isolation,
                                "regimes": regimes_summary, "inputs_sha256": input_hashes,
                                "planted_1ulp": planted_check(expect["rthcuten"][:nz])})
            if any(v["differing_words"] for v in fields.values()) or any(
                    v["differing_words"] for v in checked_fields.values()):
                failed = True
    backend = ("cpu-native-tan-control" if args.native_tan_control else "cpu-transcription") if args.host else "gpu-production"
    result = {"pass": not failed, "backend": backend,
              "GPUWM_WRF_EXACT": os.environ.get("GPUWM_WRF_EXACT", "0"),
              "gpu_parity_proven": not args.host and not failed,
              "synthetic_columns_per_mode": 288, "configurations": 12,
              "row_widths": list(ROW_WIDTHS) if not args.host else [],
              **libm_identity(), "results": all_results}
    args.receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}))
    for r in all_results:
        print(r["nz"], r["config"], {n: (v["differing_words"],v["max_ulp"]) for n,v in r["fields"].items()})
        for width, fields in r["row_batched"].items():
            print("  rows of", width, {n: v["differing_words"] for n, v in fields.items()})
    return int(failed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--host", action="store_true")
    group.add_argument("--gpu", action="store_true")
    p.add_argument("--native-tan-control", action="store_true")
    p.add_argument("--unsafe-top", action="store_true", help="reproduce the original cuflx bounds error with the checked oracle")
    args = p.parse_args()
    args.build = args.build.resolve()
    if args.native_tan_control and not args.host:
        p.error("--native-tan-control is a CPU diagnostic, not a GPU implementation")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
