"""Witness native trig rounding on real CAM inputs without changing the referee.

The host adapter must contain the production MIT CORE-MATH cos/acos functions.
The native executable and deck are the recorded standalone replay of captured
WRF columns. Both the original and correctly rounded trajectories are traced.
System MPFR is a development-only independent oracle; no MPFR or glibc code is
copied into the engine. Raw native differences remain visible in the receipt.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import datetime
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess

import numpy as np


TRACE_WRAPPERS = r'''
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cfenv>
#include <dlfcn.h>
static double witness_trig(unsigned long long kind, double x, double cr) {
  if (fegetround() != FE_TONEAREST) std::abort();
  static void *libm = dlopen("libm.so.6", RTLD_NOW | RTLD_LOCAL);
  static auto real_cos = reinterpret_cast<double(*)(double)>(dlsym(libm,"cos"));
  static auto real_acos = reinterpret_cast<double(*)(double)>(dlsym(libm,"acos"));
  if (!real_cos || !real_acos) std::abort();
  double native = (kind == 0 ? real_cos : real_acos)(x);
  static FILE *trace = std::fopen(std::getenv("CAM5_TRIG_TRACE"),"wb");
  if (!trace) std::abort();
  unsigned long long row[4] = {kind,0,0,0};
  std::memcpy(row+1,&x,8); std::memcpy(row+2,&native,8);
  std::memcpy(row+3,&cr,8);
  if (std::fwrite(row,sizeof(row),1,trace) != 1) std::abort();
  return std::getenv("CAM5_TRIG_CORRECTLY_ROUNDED") ? cr : native;
}
extern "C" double cos(double x) noexcept {
  return witness_trig(0,x,g64_cos::cr_cos(x));
}
extern "C" double acos(double x) noexcept {
  return witness_trig(1,x,g64_acos::cr_acos(x));
}
'''


class MPFRValue(ct.Structure):
    _fields_ = [("precision", ct.c_long), ("sign", ct.c_int),
                ("exponent", ct.c_long), ("limbs", ct.POINTER(ct.c_ulong))]


class Bounds:
    """Directed MPFR bounds, then identical binary64 ties-to-even endpoints."""
    def __init__(self):
        self.lib = ct.CDLL("libmpfr.so.6")
        ptr = ct.POINTER(MPFRValue)
        signatures = {
            "mpfr_init2": ([ptr, ct.c_long], None),
            "mpfr_clear": ([ptr], None),
            "mpfr_set_d": ([ptr, ct.c_double, ct.c_int], ct.c_int),
            "mpfr_cos": ([ptr, ptr, ct.c_int], ct.c_int),
            "mpfr_acos": ([ptr, ptr, ct.c_int], ct.c_int),
            "mpfr_get_d": ([ptr, ct.c_int], ct.c_double),
            "mpfr_get_str": ([ct.c_void_p, ct.POINTER(ct.c_long), ct.c_int,
                              ct.c_size_t, ptr, ct.c_int], ct.c_void_p),
            "mpfr_free_str": ([ct.c_void_p], None),
            "mpfr_get_version": ([], ct.c_char_p),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.lib, name); fn.argtypes = args; fn.restype = result

    def hexadecimal(self, value):
        exponent = ct.c_long()
        text = self.lib.mpfr_get_str(None, ct.byref(exponent), 16, 0,
                                     ct.byref(value), 0)
        digits = ct.string_at(text).decode()
        self.lib.mpfr_free_str(text)
        return {"digits": digits, "exponent": exponent.value,
                "value_rule": "digits * 16**(exponent - number_of_unsigned_digits)"}

    def rounded(self, kind, x, *, include_bounds):
        for precision in (256, 512, 1024):
            values = [MPFRValue() for _ in range(3)]
            for v in values: self.lib.mpfr_init2(ct.byref(v), precision)
            argument, lower, upper = values
            try:
                assert self.lib.mpfr_set_d(ct.byref(argument), x, 0) == 0
                fn = self.lib.mpfr_cos if kind == 0 else self.lib.mpfr_acos
                fn(ct.byref(lower), ct.byref(argument), 3)  # MPFR_RNDD
                fn(ct.byref(upper), ct.byref(argument), 2)  # MPFR_RNDU
                lo = self.lib.mpfr_get_d(ct.byref(lower), 0)  # MPFR_RNDN
                hi = self.lib.mpfr_get_d(ct.byref(upper), 0)
                if bits(lo) == bits(hi):
                    detail = {"precision_bits": precision}
                    if include_bounds:
                        detail.update(lower=self.hexadecimal(lower),
                                      upper=self.hexadecimal(upper),
                                      lower_rounded_hex=lo.hex(), upper_rounded_hex=hi.hex())
                    return bits(lo), detail
            finally:
                for v in values: self.lib.mpfr_clear(ct.byref(v))
        raise AssertionError("directed bounds did not determine binary64 rounding")


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def value(word):
    return struct.unpack("<d", struct.pack("<Q", int(word)))[0]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-exe", type=Path, required=True)
    p.add_argument("--deck", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--native-reference", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    assert os.environ.get("GPUWM_NUMERICS") == "strict"
    assert os.environ.get("GPUWM_WRF_EXACT") == "1"
    a.work.mkdir(parents=True, exist_ok=True)
    source = a.adapter.read_text()
    for name in ("cos", "acos"):
        wrapper = ('extern "C" double ' + name + '(double x) noexcept '
                   '{return g64_' + name + '::cr_' + name + '(x);}')
        assert source.count(wrapper) == 1
        source = source.replace(wrapper, "")
    assert "Permission is hereby granted" in source
    adapter = a.work / "trace-core-math.cpp"
    adapter.write_text(source + TRACE_WRAPPERS)
    library = a.work / "trace-core-math.so"
    commands = [["g++", "-shared", "-fPIC", "-O2", "-fno-fast-math",
                 "-ffp-contract=off", str(adapter), "-o", str(library), "-lm", "-ldl"]]
    subprocess.run(commands[-1], check=True)
    records = []; outputs = {}
    for mode in ("native", "correctly-rounded"):
        trace = a.work / (mode + "-calls.bin")
        output = a.work / (mode + "-output.bin")
        env = dict(os.environ, LD_PRELOAD=str(library), CAM5_TRIG_TRACE=str(trace))
        env.pop("CAM5_TRIG_CORRECTLY_ROUNDED", None)
        if mode == "correctly-rounded": env["CAM5_TRIG_CORRECTLY_ROUNDED"] = "1"
        cmd = [str(a.native_exe), str(a.deck), str(output), "unused", "unused"]
        commands.append({"argv": cmd, "trace": str(trace), "trig_mode": mode})
        subprocess.run(cmd, env=env, cwd=a.work, check=True)
        records.append(np.fromfile(trace, dtype="<u8").reshape(-1, 4))
        outputs[mode] = output
    assert sha(outputs["native"]) == sha(a.native_reference), "tracing changed native output"
    all_calls = np.concatenate(records)
    unique = np.unique(all_calls, axis=0)
    arguments = np.unique(all_calls[:, :2], axis=0)
    oracle = Bounds(); witnesses = []; expected = []
    for kind, argument in arguments:
        rows = unique[(unique[:, 0] == kind) & (unique[:, 1] == argument)]
        assert len(rows) == 1, "same trig input produced inconsistent output"
        _, _, native, cr = map(int, rows[0])
        correct, detail = oracle.rounded(int(kind), value(argument), include_bounds=native != cr)
        assert cr == correct, "production CORE-MATH host adaptation is not correctly rounded"
        expected.append(correct)
        if native != correct:
            assert abs(native - correct) == 1, "unexpected native error exceeds one ULP"
            witnesses.append({"function": "cos" if kind == 0 else "acos",
                              "input_bits": f"0x{int(argument):016x}",
                              "input_hex": value(argument).hex(),
                              "native_bits": f"0x{native:016x}", "native_hex": value(native).hex(),
                              "correct_bits": f"0x{correct:016x}", "correct_hex": value(correct).hex(),
                              "native_ulp_error": 1, "certificate": detail,
                              "occurrences": int(np.sum((all_calls[:, 0] == kind) &
                                                        (all_calls[:, 1] == argument)))})
    # Run the actual production CUDA helpers and complete captured-column replay.
    import cupy as cp
    from woof.core.kernels import module_source, module_options
    from tools.cammgmp_wrf461_oracle.oracle_io import read_deck, read_outputs
    from tools.cammgmp_wrf461_oracle.compare_gpu import run_group, grade
    free, total = cp.cuda.runtime.memGetInfo()
    assert free >= 1024**3, "wait for at least 1 GiB free card memory"
    production = module_source("cammgmp")
    extra = r'''
extern "C" __global__ void cam5_rounding_probe(const unsigned long long *a,
                                             double *out, int count) {
 int i=blockDim.x*blockIdx.x+threadIdx.x;
 if(i<count) { double x=__longlong_as_double(a[2*i+1]);
              out[i]=a[2*i] ? uw_acos(x) : uw_cos(x); }
}
'''
    module = cp.RawModule(code=production + extra, options=module_options("cammgmp"))
    device_args = cp.asarray(arguments); device_result = cp.empty(len(arguments), cp.float64)
    module.get_function("cam5_rounding_probe")(((len(arguments)+63)//64,), (64,),
                                             (device_args, device_result, np.int32(len(arguments))))
    got = cp.asnumpy(device_result).view(np.uint64)
    assert np.array_equal(got, np.asarray(expected, dtype=np.uint64)), "CUDA differs from bounded result"
    gpu = [run_group(g) for g in read_deck(a.deck)]
    grades = {}
    for mode, output in outputs.items():
        rows = grade(read_outputs(output), gpu)
        grades[mode] = {"words": sum(v["words"] for v in rows.values()),
                        "different_words": sum(v["differ"] for v in rows.values()),
                        "fields": {n: v for n, v in rows.items() if v["differ"]}}
    assert grades["native"]["different_words"] > 0, "this witness requires a native rounding residue"
    assert grades["correctly-rounded"]["different_words"] == 0
    assert witnesses
    report = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "kind": "native-forecast trig rounding witness", "strict_numerics": True,
              "raw_native_parity_pass": False,
              "qualified_under_correctly_rounded_math_ruling": True,
              "no_glibc_source_copied": True, "no_fixture_lookup_in_production": True,
              "mpfr_version": oracle.lib.mpfr_get_version().decode(),
              "mpfr_role": "system development oracle only; no code or runtime dependency shipped",
              "rounding_method": "directed high-precision bounds both round to the same binary64 ties-to-even word",
              "mpfr_manual": "https://mpfr.org/mpfr-current/mpfr.html#Rounding",
              "captured_columns": sum(g["ncol"] for g in read_deck(a.deck)),
              "calls_by_trajectory": {n: len(v) for n, v in zip(outputs, records)},
              "unique_trig_arguments": len(arguments), "cuda_vs_bounded_result_differences": 0,
              "native_inexact_unique_arguments": len(witnesses), "witnesses": witnesses,
              "replay": grades, "commands": commands,
              "production_module_sha256": hashlib.sha256(production.encode()).hexdigest(),
              "inputs": {str(q): sha(q) for q in (a.native_exe, a.deck, a.adapter, a.native_reference)},
              "artifacts": {str(q): {"bytes": q.stat().st_size, "sha256": sha(q)}
                            for q in [adapter, *outputs.values(),
                                      a.work / "native-calls.bin", a.work / "correctly-rounded-calls.bin"]},
              "gpu_free_bytes_before": free, "gpu_total_bytes": total}
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("witnesses", "commands", "artifacts", "inputs")}))


if __name__ == "__main__":
    main()
