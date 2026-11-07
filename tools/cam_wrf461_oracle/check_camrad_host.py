"""Execute the CUDA column source on a CPU to check indexing and arithmetic.

This checks the generated source and bounded scratch reuse without a GPU.
It is neither a runtime fallback nor evidence of GPU execution. The separate
GPU checker and a real-input native capture remain required for qualification.
"""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import subprocess

import numpy as np

from check_camrad_cpu import PARAMS
from make_driver import arguments
from validate import compare, load


SHIM = r'''
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#define __device__
#define __global__
#define __forceinline__ inline
#define __launch_bounds__(n)
struct Dim { int x; };
static thread_local Dim blockIdx, blockDim, threadIdx, gridDim;
static inline double __longlong_as_double(unsigned long long x) {
    double y; std::memcpy(&y, &x, 8); return y;
}
static inline int __clzll(unsigned long long x) {
    return x ? __builtin_clzll(x) : 64;
}
static inline unsigned long long __umul64hi(unsigned long long x,
                                           unsigned long long y) {
    return (static_cast<unsigned __int128>(x) * y) >> 64;
}
static inline double __dadd_rn(double x,double y){return x+y;}
static inline double __dsub_rn(double x,double y){return x-y;}
static inline double __dmul_rn(double x,double y){return x*y;}
static inline double __ddiv_rn(double x,double y){return x/y;}
static inline double __dsqrt_rn(double x){return std::sqrt(x);}
static inline double __fma_rn(double x,double y,double z){return std::fma(x,y,z);}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("build", type=Path)
    parser.add_argument("--levels", type=int, default=48)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1 or args.workers % 8:
        parser.error("--workers must be a positive multiple of 8")
    root = Path(__file__).resolve().parents[2]
    kdir = root / "woof/core/kernels"
    metadata = json.loads((kdir / "cam3_emissivity.json").read_text())["camrad"]
    fields = metadata["parameters"]
    casts = ",".join(f"({f['dtype']}*)buffers[{i}]" for i, f in enumerate(fields))
    wrapper = '''
extern "C" void cam_host(int columns,int workers,const unsigned char* state,
                         unsigned char* workspace,void** buffers) {
    #pragma omp parallel for num_threads(2) schedule(static)
    for(int worker=0;worker<workers;++worker) {
        blockIdx.x=worker/8;threadIdx.x=worker%8;
        blockDim.x=8;gridDim.x=workers/8;
        cam_driver(columns,state,workspace,CASTS);
    }
}
'''.replace("CASTS", casts)
    source = (kdir / "cam3_emissivity.cu").read_text()
    source = source.replace('asm("trap;");', "__builtin_trap();")
    args.build.mkdir(parents=True, exist_ok=True)
    cpp = args.build.resolve() / "cam_host.cpp"
    library = args.build.resolve() / "cam_host.so"
    cpp.write_text(SHIM + (kdir / "glibc_flt64.cuh").read_text() + source + wrapper)
    command = ["g++", "-std=c++17", "-O2", "-fPIC", "-shared", "-fopenmp",
               "-fno-fast-math", "-ffp-contract=off", f"-DCAM_LEVELS={args.levels}",
               str(cpp), "-o", str(library)]
    subprocess.run(command, check=True)
    data = load(args.oracle)
    _, declarations = arguments(args.source.read_text(), "camrad")
    params = {**PARAMS, "kte": args.levels, "kde": args.levels+1,
              "kme": args.levels+1, "cam_abs_dim2": args.levels+1}
    buffers = []
    for field in fields:
        name = field["name"]
        dtype = {"int": np.int32, "float": np.float32, "double": np.float64}[field["dtype"]]
        if name in params:
            array = np.full((64, 1), params[name], dtype)
        else:
            role = "input" if ("camrad", 1, "input:"+name) in data else "output"
            array = np.stack([data[("camrad", i, role+":"+name)] for i in range(1, 65)])
            if name != "gases" and "ims:ime" in declarations[name]:
                array = array[:, ::2]
            array = array.astype(dtype) if role == "input" else np.zeros(array.shape, dtype)
        count = eval(field.get("count_expression", str(field["count"])),
                     {"__builtins__": {}}, {"CAM_LEVELS": args.levels})
        if array.shape != (64, count):
            raise ValueError(f"{name}: invalid shape {array.shape}, expected (64,{count})")
        buffers.append(np.ascontiguousarray(array))
    work_bytes = eval(metadata["workspace_expression"], {"__builtins__": {}},
                      {"CAM_LEVELS": args.levels})
    # Guard the allocation to catch a worker/column stride mix-up.
    workspace = np.full(args.workers*work_bytes+32, 0xA5, np.uint8)
    state = np.frombuffer((args.oracle / "state.bin").read_bytes(), np.uint8)
    pointers = (ctypes.c_void_p*len(buffers))(*(x.ctypes.data for x in buffers))
    lib = ctypes.CDLL(str(library))
    lib.cam_host.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.cam_host(64, args.workers, state.ctypes.data, workspace.ctypes.data+16, pointers)
    guards = bool(np.all(workspace[:16] == 0xA5) and np.all(workspace[-16:] == 0xA5))
    expected, actual = {}, {}
    for field, array in zip(fields, buffers):
        name = field["name"]
        if ("camrad", 1, "output:"+name) not in data:
            continue
        for i in range(1, 65):
            key = ("camrad", i, "output:"+name)
            reference = data[key]
            expected[key] = reference[::2] if "ims:ime" in declarations[name] else reference
            actual[key] = array[i-1].astype(np.float64)
    grade = compare(expected, actual)
    receipt = {"execution": "CPU emulation of CUDA source, not GPU proof",
               "oracle_inputs": "analytic strict WRF columns, not a forecast capture",
               "command": command, "levels": args.levels, "workers": args.workers,
               "workspace_guards_intact": guards, "comparison": grade,
               "pass": guards and grade["pass"]}
    args.out.write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({"pass": receipt["pass"], "words": grade["words"],
                      "failed_fields": len(grade["mismatches"]), "guards": guards}))
    raise SystemExit(not receipt["pass"])


if __name__ == "__main__":
    main()
