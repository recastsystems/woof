"""Module replay (measurement tool, not product code): rerun captured MYNN-EDMF calls on two module files.

Each capture folder (capture_mynn.py) holds every device array of one MynnEdmfCuda.run call and its scalars.
Both modules run the same call from fresh copies of those inputs; every buffer (all intent in/out/inout
arrays, aliases kept) and the per-column error words are then compared word for word. Each module runs the
call twice (repeat check); the second run is timed with CUDA events around run().
Usage: replay_mynn.py --reference OLD.py --candidate NEW.py --numerics strict|fast --out OUT.json CAPTURE_DIR...
--reference-image/--candidate-image load a physics-resources packet image instead of compiling (the image a
run loads); WOOF_HEX_MYNNEDMF_LANES forces the candidate's lane width (1, 4 or 16).
"""
import argparse
import importlib.util
import json
import os
import time

import numpy as np


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(folder, manifest, cp):
    buffers = {}
    for row in manifest["arrays"]:
        if row["buffer"] not in buffers:
            host = np.load(os.path.join(folder, f"b{row['buffer']:03d}.npy"))
            buffers[row["buffer"]] = cp.array(host, order=row["order"])
    arguments = {row["name"]: buffers[row["buffer"]] for row in manifest["arrays"]}
    arguments.update(manifest["scalars"])
    return arguments, buffers


def run(backend, folder, manifest, cp):
    arguments, buffers = build(folder, manifest, cp)
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    cp.cuda.Device().synchronize()
    start.record()
    result = backend.run(arguments)
    stop.record()
    stop.synchronize()
    out = {i: cp.asnumpy(b) for i, b in buffers.items()}
    out["errflg"] = cp.asnumpy(result["errflg"])
    return out, cp.cuda.get_elapsed_time(start, stop)


def words(a):
    a = np.ascontiguousarray(a)
    return a.view({1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[a.dtype.itemsize]).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--numerics", choices=("strict", "fast"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference-image", help="precompiled image (physics-resources packet) instead of compiling")
    ap.add_argument("--candidate-image", help="precompiled image (physics-resources packet) instead of compiling")
    ap.add_argument("captures", nargs="+")
    args = ap.parse_args()
    from woof.hex import numerics
    numerics.set_mode(args.numerics, source="replay_mynn")
    import cupy as cp
    report = {"numerics": args.numerics, "reference": args.reference, "candidate": args.candidate, "calls": []}
    modules = {}
    for label, path in (("reference", args.reference), ("candidate", args.candidate)):
        module = load_module("mynn_" + label, path)
        image = getattr(args, label + "_image")
        backend = module.MynnEdmfCuda(compiled_image=open(image, "rb").read() if image else None)
        report[label + "_image"] = image
        t = time.perf_counter()
        backend._compile()
        modules[label] = backend
        report[label + "_compile_s"] = round(time.perf_counter() - t, 1)
        from cupy.cuda import driver
        kernels = getattr(backend, "_kernels", None) or {1: backend._kernel}
        report[label + "_kernel"] = {f"l{w}": {k: int(driver.funcGetAttribute(a, f.ptr))
                                     for k, a in (("num_regs", 4), ("local_size_bytes", 3), ("max_threads_per_block", 0))}
                                     for w, f in kernels.items()}
        print(label, report[label + "_compile_s"], "s compile", report[label + "_kernel"], flush=True)
    for folder in args.captures:
        manifest = json.load(open(os.path.join(folder, "manifest.json")))
        names = {}
        for row in manifest["arrays"]:
            names.setdefault(row["buffer"], row["name"])
        names["errflg"] = "errflg"
        inputs = {i: np.load(os.path.join(folder, f"b{i:03d}.npy")) for i in range(manifest["buffers"])}
        ref1, _ = run(modules["reference"], folder, manifest, cp)
        ref2, ref_ms = run(modules["reference"], folder, manifest, cp)
        cand1, _ = run(modules["candidate"], folder, manifest, cp)
        cand2, cand_ms = run(modules["candidate"], folder, manifest, cp)
        row = {"call": os.path.basename(folder), "compared_words": 0, "different_words": 0, "written_words": 0,
               "reference_repeat_different": 0, "candidate_repeat_different": 0, "differing_buffers": {},
               "reference_ms": round(ref_ms, 3), "candidate_ms": round(cand_ms, 3),
               "reference_batch_columns": getattr(modules["reference"], "last_batch_columns", None),
               "candidate_batch_columns": getattr(modules["candidate"], "last_batch_columns", None), "candidate_lane_width": getattr(modules["candidate"], "last_lane_width", None)}
        for i in ref1:
            a, b = words(ref1[i]), words(cand1[i])
            diff = int(np.count_nonzero(a != b))
            row["compared_words"] += a.size
            row["different_words"] += diff
            if i in inputs:
                row["written_words"] += int(np.count_nonzero(a != words(inputs[i])))
            row["reference_repeat_different"] += int(np.count_nonzero(a != words(ref2[i])))
            row["candidate_repeat_different"] += int(np.count_nonzero(b != words(cand2[i])))
            if diff:
                x, y = ref1[i].ravel().astype(np.float64), cand1[i].ravel().astype(np.float64)
                with np.errstate(invalid="ignore"):
                    rel = np.abs(x - y) / np.maximum(np.abs(x), 1e-30)
                row["differing_buffers"][names[i]] = {"words": diff, "of": int(a.size),
                                                       "max_abs": float(np.nanmax(np.abs(x - y))),
                                                       "max_rel": float(np.nanmax(rel))}
        report["calls"].append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "differing_buffers"}), flush=True)
        for name, d in row["differing_buffers"].items():
            print("   ", name, d, flush=True)
    report["all_bitwise"] = all(r["different_words"] == 0 for r in report["calls"])
    with open(args.out, "w") as stream:
        json.dump(report, stream, indent=1)
    print("ALL BITWISE" if report["all_bitwise"] else "DIFFERENT", flush=True)


if __name__ == "__main__":
    main()
