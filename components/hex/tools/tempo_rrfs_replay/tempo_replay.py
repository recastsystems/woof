"""Module replay (measurement tool, not product code): rerun captured TEMPO slot calls on two module files.

Each capture folder (capture_launch.py) holds every device buffer the call's pool dict reaches, its
scalars and the launch keywords. Both modules run the same call from fresh copies of those inputs;
every buffer is then compared word for word. A second run of each module is timed with CUDA events.
Usage: tempo_replay.py --reference OLD.py --candidate NEW.py --numerics strict|fast CAPTURE_DIR...
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(folder, manifest, cp):
    buffers = {i: cp.asarray(np.load(os.path.join(folder, f"b{i:03d}.npy"))) for i in range(manifest["buffers"])}
    p = {}

    def put(path, value):
        node = p
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value

    for row in manifest["arrays"]:
        put(row["path"], buffers[row["buffer"]])
    for row in manifest["scalars"]:
        put(row["path"], row["value"])
    return p, buffers


def run(module, folder, manifest, cp, timed=False):
    p, buffers = build(folder, manifest, cp)
    kw = dict(manifest["keywords"])
    kw["flags"] = tuple(bool(v) for v in kw["flags"])
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    cp.cuda.Device().synchronize()
    start.record()
    module.launch_mpas(p, **kw)
    stop.record()
    stop.synchronize()
    ms = cp.cuda.get_elapsed_time(start, stop)
    return {i: cp.asnumpy(b) for i, b in buffers.items()}, ms


def words(a):
    a = np.ascontiguousarray(a)
    return a.view({1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[a.dtype.itemsize]).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--numerics", choices=("strict", "fast"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("captures", nargs="+")
    args = ap.parse_args()
    from woof.hex import numerics
    numerics.set_mode(args.numerics, source="tempo_replay")
    import cupy as cp
    reference = load_module("tempo_reference", args.reference)
    candidate = load_module("tempo_candidate", args.candidate)
    report = {"numerics": args.numerics, "reference": args.reference, "candidate": args.candidate, "calls": []}
    for folder in args.captures:
        manifest = json.load(open(os.path.join(folder, "manifest.json")))
        names = {}
        for row in manifest["arrays"]:
            names.setdefault(row["buffer"], "/".join(map(str, row["path"])))
        inputs = {i: np.load(os.path.join(folder, f"b{i:03d}.npy")) for i in range(manifest["buffers"])}
        ref1, _ = run(reference, folder, manifest, cp)
        ref2, ref_ms = run(reference, folder, manifest, cp)
        cand1, _ = run(candidate, folder, manifest, cp)
        cand2, cand_ms = run(candidate, folder, manifest, cp)
        row = {"call": os.path.basename(folder), "compared_words": 0, "different_words": 0,
               "written_words": 0, "reference_repeat_different": 0, "candidate_repeat_different": 0,
               "differing_buffers": {}, "reference_ms": round(ref_ms, 3), "candidate_ms": round(cand_ms, 3)}
        for i in sorted(ref1):
            a, b = words(ref1[i]), words(cand1[i])
            diff = int(np.count_nonzero(a != b))
            row["compared_words"] += a.size
            row["different_words"] += diff
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
