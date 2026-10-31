"""Offline compile and pool-only timing of exact TEMPO source variants.

Compile uses NVRTC directly with an explicit SM, without opening a GPU context.
Images are supplied to the unmodified module loader during replay. This separates
CPU compilation from the pool lease and retains verbose ptxas spill evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

import numpy as np

from tempo_replay import load_module


def source(module, maths, levels, shape="warp"):
    # Exact lane proof excludes the separately owned fast arithmetic transform.
    import inspect
    kw = {"max_levels": levels}
    builder = module.warp_source if shape == "warp" else module.cuda_source
    if "numerics" in inspect.signature(builder).parameters:
        kw["numerics"] = "strict"
    return builder(maths, **kw)


def compile_image(args):
    from woof.hex import numerics
    from woof.hex.mpasphys import resources_build as resources
    from cupy_backends.cuda.libs import nvrtc
    module = load_module("tempo_compile", args.module)
    maths = resources.tempo_maths(resources._vendored_maths())
    text = source(module, maths, args.levels, args.shape)
    options = numerics.options(module.COMPILE_OPTIONS, numerics=args.numerics)
    options += ("--gpu-architecture=sm_120", "--device-as-default-execution-space", "--ptxas-options=-v")
    program = nvrtc.createProgram(text, "kern.cu", (), ())
    try:
        nvrtc.compileProgram(program, options)
        Path(args.out).write_bytes(nvrtc.getCUBIN(program))
    finally:
        Path(args.out + ".ptxas.log").write_text(nvrtc.getProgramLog(program).rstrip("\0"))
        nvrtc.destroyProgram(program)
    Path(args.out + ".json").write_text(json.dumps({
        "module": args.module, "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "numerics": args.numerics, "options": options, "levels": args.levels,
        "exact_source": True, "shape": args.shape}, indent=2))


def neighbours():
    return subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                          capture_output=True, text=True, timeout=10).stdout


def bind(manifest, buffers):
    p = {}
    for row in manifest["arrays"] + manifest["scalars"]:
        node = p
        for key in row["path"][:-1]:
            node = node.setdefault(key, {})
        node[row["path"][-1]] = buffers[row["buffer"]] if "buffer" in row else row["value"]
    return p


def replay(args):
    from woof.hex import numerics
    numerics.set_mode(args.numerics, source="tempo occupancy replay")
    import cupy as cp
    from cupy.cuda import driver
    from woof.hex.cuda_backend import aot
    import inspect
    import time
    modules = []
    module_receipts = []
    images = {}
    for number, path in enumerate(args.modules):
        module = load_module("tempo_variant_" + str(number), path)
        image_path = Path(args.images) / (Path(path).stem + "-" + args.numerics + ".cubin")
        meta = json.loads(Path(str(image_path) + ".json").read_text())
        if meta['numerics'] != args.numerics:
            raise ValueError('image numerics mismatch')
        images[meta["source_sha256"]] = image_path.read_bytes()
        original = module.warp_source
        if "numerics" in inspect.signature(original).parameters:
            module.warp_source = lambda maths, _fn=original, **kw: _fn(maths, **kw, numerics="strict")
        modules.append((Path(path).stem, module))
        module_receipts.append({"name": Path(path).stem, "path": path,
                                "source_sha256": meta["source_sha256"],
                                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                                "options": meta["options"]})
    def image_for(text, options, arch):
        return images[hashlib.sha256(text.encode()).hexdigest()]
    aot.direct_image = image_for
    report = {"numerics": args.numerics, "exact_source": True,
              "modules": module_receipts, "captures": args.captures,
              "columns_requested": args.columns, "repeat": args.repeat,
              "neighbours_start": neighbours(), "calls": []}

    def count_different(actual, compact, column_ids, factor):
        counts = []
        for i, b in actual.items():
            a = compact[i]
            dtype = {1: cp.uint8, 2: cp.uint16, 4: cp.uint32, 8: cp.uint64}[b.dtype.itemsize]
            if isinstance(a, np.ndarray):
                # Immutable tables use the pristine host capture as reference.
                # Check every word in bounded chunks, without a second 1 GB
                # device copy competing with the tiled state for pool capacity.
                chunks = []
                flat_a, flat_b = a.reshape(-1), b.reshape(-1)
                chunk = (64 * 1024 * 1024) // a.dtype.itemsize
                for start in range(0, a.size, chunk):
                    expected = cp.asarray(flat_a[start:start+chunk])
                    chunks.append(cp.count_nonzero(flat_b[start:start+chunk].view(dtype) != expected.view(dtype)))
                counts.append(cp.sum(cp.stack(chunks)))
            else:
                if i in column_ids and factor > 1:
                    b = b.reshape(a.shape[:-1] + (factor, a.shape[-1]))
                    a = a[..., None, :]
                counts.append(cp.count_nonzero(b.view(dtype) != a.view(dtype)))
        # One tiny host read for the proof, outside every timing interval.
        return [int(x) for x in cp.asnumpy(cp.stack(counts))]

    for folder in args.captures:
        began = time.monotonic()
        manifest = json.loads((Path(folder) / "manifest.json").read_text())
        host = {i: np.load(Path(folder) / f"b{i:03d}.npy") for i in range(manifest["buffers"])}
        base_p = bind(manifest, host)
        cells = base_p["fields"]["qv"].shape[-1]
        nz = base_p["fields"]["qv"].shape[0]
        factor = max(1, (args.columns + cells - 1) // cells)
        if factor > 1 and base_p["first_call_main"]:
            raise ValueError("first_call_main is global-column-zero state; tile a warm call (12 or 24)")
        column_ids = {row["buffer"] for row in manifest["arrays"]
                      if row["path"][0] != "tables" and host[row["buffer"]].ndim
                      and host[row["buffer"]].shape[-1] == cells}
        kw = dict(manifest["keywords"])
        kw["flags"] = tuple(map(bool, kw["flags"]))
        reference = {i: cp.asarray(b) for i, b in host.items()}
        modules[0][1].launch_mpas(bind(manifest, reference), **kw)
        cp.cuda.get_current_stream().synchronize()
        for _, module in modules:
            module._WARP_WORK.clear()
        table_ids = {row["buffer"] for row in manifest["arrays"] if row["path"][0] == "tables"}
        table_reference = {i: host[i] for i in table_ids}
        table_device = {i: reference[i] for i in table_ids}
        if any(count_different(table_device, table_reference, set(), 1)):
            raise AssertionError("baseline mutated the immutable tables")
        work = {i: cp.tile(cp.asarray(b), (1,) * (b.ndim-1) + (factor,))
                if i in column_ids else (table_device[i] if i in table_ids else cp.asarray(b))
                for i, b in host.items()}
        reference.update(table_reference)
        cp.get_default_memory_pool().free_all_blocks()
        device = cp.cuda.runtime.getDevice()
        def restore():
            for i, b in host.items():
                if i in column_ids:
                    work[i].reshape(b.shape[:-1] + (factor, cells))[...] = cp.asarray(b)[..., None, :]
                else:
                    work[i].set(b)
        def launch(module):
            p = bind(manifest, work)
            _, sizes, resident = module._compiled_warp(device, p['device_math_source'], nz)
            module._warp_work(device, nz, min(cells*factor, module.WARP_WAVES*resident), sizes)
            start, stop = cp.cuda.Event(), cp.cuda.Event()
            cp.cuda.get_current_stream().synchronize()
            start.record()
            module.launch_mpas(p, **kw)
            stop.record()
            stop.synchronize()
            value = cp.cuda.get_elapsed_time(start, stop)
            module._WARP_WORK.clear()
            return value
        rows = {}
        for name, module in modules:
            restore()
            launch(module)
            counts = count_different(work, reference, column_ids, factor)
            compact = {i: b[..., :cells].copy() if i in column_ids else
                       (reference[i] if counts[i] == 0 else b.copy()) for i, b in work.items()}
            consistency = sum(count_different(work, compact, column_ids, factor))
            restore()
            launch(module)
            repeat = sum(count_different(work, compact, column_ids, factor))
            second = sum(count_different(work, reference, column_ids, factor))
            rows[name] = {'module': name, 'call': Path(folder).name, 'columns': cells*factor,
                          'different_words': sum(counts), 'repeat_different': repeat,
                          'second_different_words': second, 'tile_consistency_different': consistency,
                          'compared_words': sum(b.size for b in work.values()),
                          'device_pool_total_bytes': cp.get_default_memory_pool().total_bytes(),
                          'differing_buffers': {str(i): c for i, c in zip(work, counts) if c}}
            if factor > 1:
                rows[name]['tiled_vs_original_different'] = sum(counts)
            del compact
            cp.get_default_memory_pool().free_all_blocks()
            print(json.dumps({'phase': 'proof', 'elapsed_s': time.monotonic()-began, **rows[name]}), flush=True)
        times = {name: [] for name, _ in modules}
        for rep in range(args.repeat):
            for name, module in modules:
                restore()
                times[name].append(launch(module))
                cp.get_default_memory_pool().free_all_blocks()
        for name, module in modules:
            mod, _, resident = module._compiled_warp(device, base_p['device_math_source'], nz)
            kernel = mod.get_function('tempo_warp_steps')
            row = rows[name]
            row.update(median_ms=statistics.median(times[name]),
                       ms_per_100000_columns=statistics.median(times[name])*100000/(cells*factor),
                       samples_ms=times[name], registers=driver.funcGetAttribute(4,kernel.ptr),
                       local_bytes=driver.funcGetAttribute(3,kernel.ptr), resident_work_slots=resident,
                       threads_per_block=module.WARP_BLOCK)
            report['calls'].append(row)
            print(json.dumps(row), flush=True)
        del reference, work, host, base_p, table_reference, table_device
        for _, module in modules:
            module._WARP_WORK.clear()
            module._ORDER_WORK.clear()
            module._POINTER_TABLES.clear()
        cp.get_default_memory_pool().free_all_blocks()
        # Retain a complete receipt even if a later call hits the wall timeout.
        Path(args.out).write_text(json.dumps(report,indent=2))
    report['neighbours_end'] = neighbours()
    report['all_bitwise'] = all(r['different_words'] == r['repeat_different'] == r['second_different_words'] == r['tile_consistency_different'] == 0 for r in report['calls'])
    Path(args.out).write_text(json.dumps(report,indent=2))
    if not report['all_bitwise']:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    comp = sub.add_parser("compile")
    comp.add_argument("--module", required=True)
    comp.add_argument("--levels", type=int, default=59)
    comp.add_argument("--shape", choices=("warp", "column"), default="warp")
    run = sub.add_parser("replay")
    run.add_argument("--modules", nargs="+", required=True)
    run.add_argument("--images", required=True)
    run.add_argument("--columns", type=int, default=2562)
    run.add_argument("--repeat", type=int, default=9)
    run.add_argument("captures", nargs="+")
    for cmd in (comp, run):
        cmd.add_argument("--out", required=True)
        cmd.add_argument("--numerics", choices=("fast", "strict"), required=True)
    args = parser.parse_args()
    if args.command == "compile":
        compile_image(args)
    else:
        replay(args)


if __name__ == "__main__":
    main()
