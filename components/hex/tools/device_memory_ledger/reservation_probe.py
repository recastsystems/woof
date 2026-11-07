#!/usr/bin/env python3
"""Measure ONE kernel's CUDA local-memory reservation, in its own process.

A kernel with a non-zero local frame takes a device-wide backing store the
first time it is launched.  Kernels reachable only through
cupy.cuda.function.Module (the direct-NVRTC route RRTMG uses) never pass a
cupy.RawModule proxy, so an in-run tracer cannot see them.

The launch here is a NULL LAUNCH: every integer scalar argument is 0 and
every pointer is a small dummy array.  These kernels open with a count
guard (``if (tid >= ncol * NGPTLW) return;``,
``if (col >= n) return;``), so with every count at zero no thread reaches a
dereference -- but the driver has already taken the reservation, which is
what is being measured.

Because a kernel that dereferences BEFORE its guard would fault the whole
context, each kernel is measured in its own process by --sweep, so one bad
kernel costs one row and not the run.

Validation is built in, and it was re-pinned 2026-08-25 (stale-guard audit,
finding 6).  The original positive control pinned ``gf_gfdrv_stage``
at 7,034.0 MiB -- the sum of the pre-cut in-run increments (rlw_rtrn_march
254.0 + ysu_column 1790.0 + gf_gfdrv_stage 4990.0).  The
Grell-Freitas frame cut retired that premise (the gf frame is now 88/72 B),
so the instrument declared its own technique invalid against a dead number.
The control is now the POST-CUT widest launched frame, ``wsm6_column`` at
7,216 B (STATE.md section 5), checked against a bound DERIVED from the live
device: 0 < reservation <= local_size_bytes x SM count x max resident
threads per SM (1,797.0 MiB on the 170 SM card).  A derived bound is weaker
than the retired measured-equality check; restoring equality needs one
post-cut control run on real hardware, which is a NAMED FOLLOW-UP -- this
re-pin ships without it because the probe cannot run without a CUDA device
and a per-process nvidia-smi row (WDDM boxes publish none).  The negative
control (a zero-local kernel must reserve nothing) is unchanged and stays
exact.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

MIB = 1024.0 * 1024.0
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from kernel_reservation import func_attributes, nvsmi_process_mib  # noqa: E402

PARAM = re.compile(r"^\s*(?:const\s+)?([A-Za-z_][A-Za-z0-9_ ]*?)\s*(\*)?\s*"
                   r"(?:__restrict__\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*$")

# module import path -> (builder attr, builder args)
BUILDERS = {
    "woof.core.rrtmg_lw": ("_gpu_module", ()),
    "woof.core.rrtmg_mcica": ("_mcica_gpu_module", ()),
    "woof.core.gf": ("_gf_module", (55,)),
    # The registry route: woof.core.kernels.load_module("wsm6") is how the
    # engine builds the module holding the POST-CUT widest launched frame
    # (wsm6_column, 7,216 B -- STATE.md section 5).  The per-module builder
    # attrs above cannot reach it because wsm6 has no module-level builder.
    "woof.core.kernels": ("load_module", ("wsm6",)),
}


def load_probe_module(module_path):
    """Return the compiled module and the source used by its loader."""
    if module_path == "woof.core.ysu":
        from woof import wrf_exact, ysu_generations
        from woof.core import kernels

        name, defines = ysu_generations.ysu_kernel_selection(
            ysu_generations.YSU_DEFAULT_GENERATION, strict=wrf_exact.ENABLED)
        module = (kernels.load_module_int_defines(name, defines) if defines
                  else kernels.load_module(name))
        return module, kernels.module_source(name)
    provider = importlib.import_module(module_path)
    builder, arguments = BUILDERS[module_path]
    return getattr(provider, builder)(*arguments), kernel_source(provider, arguments)


def kernel_source(mod, builder_args=()):
    # Registry route first: woof.core.kernels exposes module_source(name),
    # the exact string load_module hands to NVRTC.
    ms = getattr(mod, "module_source", None)
    if callable(ms) and builder_args:
        try:
            src = ms(builder_args[0])
        except Exception:
            src = None
        if isinstance(src, str):
            return src
    for attr in ("_gpu_source", "_source", "_mcica_gpu_source",
                 "_module_source", "_kernel_source"):
        fn = getattr(mod, attr, None)
        if callable(fn):
            for call in ((), (55,)):
                try:
                    src = fn(*call)
                except Exception:
                    continue
                if isinstance(src, str):
                    return src
    kdir = Path(mod.__file__).parent / "kernels"
    stem = mod.__name__.rsplit(".", 1)[-1]
    parts = []
    for path in sorted(kdir.glob(stem + "*.cu")):
        parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)


def signature(src, name):
    # Comments must go FIRST.  These signatures carry a shape comment after
    # every argument -- "// (ncol, NGPTLW)" -- and the closing paren inside
    # one of those comments captures a greedy parameter-list match one paren
    # too far, which silently corrupts only the LAST argument.
    src = re.sub(r"//[^\n]*", "", src)
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    m = re.search(r'extern\s+"C"\s+__global__\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?void\s+' + re.escape(name)
                  + r'\s*\(([^{)]*)\)\s*\{', src, re.S)
    if not m:
        m = re.search(r'__global__\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?void\s+' + re.escape(name)
                      + r'\s*\(([^{)]*)\)\s*\{', src, re.S)
    if not m:
        return None
    return [p.strip() for p in m.group(1).split(",") if p.strip()]


def build_args(cp, np, params):
    """Zero every count, dummy every pointer."""
    args = []
    dummy_f = cp.zeros(64, dtype=cp.float32)
    dummy_i = cp.zeros(64, dtype=cp.int32)
    dummy_u = cp.zeros(64, dtype=cp.uint8)
    dummy_d = cp.zeros(64, dtype=cp.float64)
    keep = [dummy_f, dummy_i, dummy_u, dummy_d]
    for p in params:
        m = PARAM.match(p)
        if not m:
            return None, None
        ctype, star, _nm = m.group(1).strip(), m.group(2), m.group(3)
        ctype = ctype.replace("__restrict__", "").strip()
        if star:
            if "char" in ctype or "uint8" in ctype:
                args.append(dummy_u)
            elif ctype.startswith("int") or ctype.endswith("int") or ctype == "int":
                args.append(dummy_i)
            elif ctype in ("double",):
                args.append(dummy_d)
            else:
                args.append(dummy_f)
        elif ctype in ("float", "real"):
            args.append(np.float32(0.0))
        elif ctype == "double":
            args.append(np.float64(0.0))
        elif "char" in ctype:
            args.append(np.uint8(0))
        else:
            args.append(np.int32(0))
    return tuple(args), keep


# The reservation is a function of the LAUNCH CONFIGURATION, not of the
# kernel alone (pre-cut example: gf_gfdrv_stage reserved 4990 MiB at its
# shipped block of 64 and 7034 MiB at 128), so every row is launched at the
# block size the shipped call site uses.
DEFAULT_SWEEP_TARGETS = (
    # POSITIVE CONTROL, re-pinned 2026-08-25 (stale-guard audit,
    # finding 6): the post-cut widest LAUNCHED frame, wsm6_column at
    # 7,216 B (STATE.md section 5), at its shipped call-site block of 32
    # (woof/core/wsm6.py::_COLUMN_TPB).  "bound" = the reservation must
    # be positive and inside the device-derived backing-store bound
    # (frame x resident threads; 1,797.0 MiB on the 170 SM card).  The
    # retired control pinned gf_gfdrv_stage at the pre-frame-cut
    # 7,034.0 MiB; the cut shrank that frame to 88/72 B, so the pin
    # failed against a dead premise.  Restoring a measured-equality
    # control needs one post-cut run on real hardware (named follow-up).
    ("woof.core.kernels", "wsm6_column", 32, "bound"),
    # The post-cut gf frames, unpinned: measured for the ledger, no
    # longer the control (their 88/72 B frames reserve ~nothing).
    ("woof.core.gf", "gf_gfdrv_stage", 64, None),
    ("woof.core.gf", "gf_deep_stage", 64, None),
    ("woof.core.gf", "gf_shallow_stage", 64, None),
    # the route no in-run proxy can see; rrtmg_lw.py:3979 launches (128,)
    ("woof.core.rrtmg_lw", "rlw_rtrn_march", 128, None),
    ("woof.core.rrtmg_lw", "rlw_cldprmc", 128, None),
    # negative control: a zero-local kernel must reserve nothing
    ("woof.core.rrtmg_mcica", None, 128, 0.0),
)


def one(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--arwen-checkout", required=True)
    p.add_argument("--module", required=True)
    p.add_argument("--kernel", required=True)
    p.add_argument("--blocks", type=int, default=4096)
    p.add_argument("--threads", type=int, default=128)
    p.add_argument("--prelaunch", default=None,
                   help="module:kernel:threads to launch FIRST.  The local "
                        "backing store is one per-context high-water pool, so "
                        "a bigger frame launched first must drive this "
                        "kernel's own increment to zero.")
    args = p.parse_args(argv)
    sys.path.insert(0, args.arwen_checkout)

    import numpy as np
    import cupy as cp

    pid = os.getpid()
    cp.cuda.runtime.free(cp.cuda.runtime.malloc(8))
    cp.cuda.runtime.deviceSynchronize()

    pre = None
    if args.prelaunch:
        pm, pk, pt = (args.prelaunch.split(":") + ["128"])[:3]
        pobj, psrc = load_probe_module(pm)
        pfn = pobj.get_function(pk)
        pargs, _pk = build_args(cp, np, signature(psrc, pk))
        b0 = nvsmi_process_mib(pid)
        pfn((4096,), (int(pt),), pargs)
        cp.cuda.runtime.deviceSynchronize()
        pre = {"kernel": pk, "threads": int(pt),
               "reservation_mib": round((nvsmi_process_mib(pid) or 0) - (b0 or 0), 1)}

    obj, src = load_probe_module(args.module)
    fn = obj.get_function(args.kernel)
    attrs = func_attributes(cp, fn)
    params = signature(src, args.kernel)
    out = {"module": args.module, "kernel": args.kernel, "attrs": attrs,
           "params": len(params or [])}
    if params is None:
        out["error"] = "signature not found in source"
        print(json.dumps(out))
        return 2
    a, _keep = build_args(cp, np, params)
    if a is None:
        out["error"] = "could not parse a parameter"
        print(json.dumps(out))
        return 2
    threads = min(args.threads, attrs.get("max_threads_per_block") or args.threads)
    cp.cuda.runtime.deviceSynchronize()
    before = nvsmi_process_mib(pid)
    fn((args.blocks,), (threads,), a)
    cp.cuda.runtime.deviceSynchronize()
    after = nvsmi_process_mib(pid)
    out["threads"] = threads
    out["blocks"] = args.blocks
    out["prelaunch"] = pre
    out["nvsmi_before_mib"] = before
    out["nvsmi_after_mib"] = after
    out["reservation_mib"] = (None if (before is None or after is None)
                              else round(after - before, 1))
    out["local_size_bytes"] = attrs.get("local_size_bytes")
    if out["local_size_bytes"]:
        out["implied_threads"] = round(
            (out["reservation_mib"] or 0) * MIB / out["local_size_bytes"])
    # The derived control bound: the backing store cannot exceed one frame
    # per device-resident thread.  On the 170 SM card and wsm6_column's
    # 7,216 B frame this is the 1,797.0 MiB upper bound STATE.md section 5
    # records; the bound is computed from the LIVE device so the control
    # stays valid on every card.
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    max_resident = (int(props["multiProcessorCount"])
                    * int(props["maxThreadsPerMultiProcessor"]))
    out["device_max_resident_threads"] = max_resident
    if out["local_size_bytes"]:
        out["derived_reservation_bound_mib"] = round(
            out["local_size_bytes"] * max_resident / MIB, 1)
    print(json.dumps(out))
    return 0


def sweep(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--arwen-checkout", required=True)
    p.add_argument("--json", default=None)
    p.add_argument("--targets", default=None,
                   help="module:kernel[,module:kernel...]; default is "
                        "DEFAULT_SWEEP_TARGETS (the wsm6 bound control, the "
                        "negative control, the gf and RRTMG frames)")
    args = p.parse_args(argv)
    default = list(DEFAULT_SWEEP_TARGETS)
    targets = []
    if args.targets:
        for t in args.targets.split(","):
            parts = t.split(":")
            targets.append((parts[0], parts[1],
                            int(parts[2]) if len(parts) > 2 else 128, None))
    else:
        targets = default
    rows = []
    for mod, kern, threads, expect in targets:
        if kern is None:
            kern = first_zero_local_kernel(args.arwen_checkout, mod)
            if kern is None:
                continue
        cmd = [sys.executable, str(HERE / "reservation_probe.py"), "one",
               "--arwen-checkout", args.arwen_checkout,
               "--module", mod, "--kernel", kern,
               "--threads", str(threads)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        try:
            rec = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:
            rec = {"module": mod, "kernel": kern,
                   "error": (r.stderr or r.stdout)[-300:]}
        rec["expected_mib"] = expect
        # A kernel the Python layer never names is never launched by the run,
        # so its frame never reaches the context's backing store.  It is
        # measured here for reference and excluded from the ledger total.
        rec["launched_from_python"] = names_in_python(args.arwen_checkout, kern)
        if expect == "bound":
            # Derived-bound control: positive, and inside the device-derived
            # backing-store bound (+2 MiB nvidia-smi granularity).
            resv = rec.get("reservation_mib")
            bound = rec.get("derived_reservation_bound_mib")
            rec["matches_expected"] = (
                resv is not None and bound is not None
                and 0.0 < resv <= bound + 2.0
            )
        elif expect is not None and rec.get("reservation_mib") is not None:
            rec["matches_expected"] = abs(rec["reservation_mib"] - expect) <= 2.0
        rows.append(rec)
    out = {"rows": rows}
    ctrl = [r for r in rows if r.get("expected_mib") is not None]
    out["controls_pass"] = bool(ctrl) and all(r.get("matches_expected")
                                              for r in ctrl)
    print("%-26s %-24s %8s %8s %10s %10s %8s"
          % ("module", "kernel", "block", "local B", "resv MiB", "expect", "ok"))
    for r in rows:
        print("%-26s %-24s %8s %8s %10s %10s %8s  %s"
              % (r.get("module", "")[-26:], r.get("kernel", "")[:24],
                 r.get("threads"), r.get("local_size_bytes"),
                 r.get("reservation_mib"), r.get("expected_mib"),
                 r.get("matches_expected", ""),
                 ("" if r.get("launched_from_python", True)
                  else "NEVER LAUNCHED, excluded ")
                 + (r.get("error") or "")[:60]))
    print("CONTROLS", "PASS" if out["controls_pass"] else "FAIL")
    if args.json:
        open(args.json, "w").write(json.dumps(out, indent=1))
    return 0 if out["controls_pass"] else 1


def names_in_python(checkout, kernel):
    # The SHIPPED package only.  tests/ and tools/ name kernels that the
    # forecast never launches (gf_deep_stage lives in a parity oracle), and
    # counting those would add a 6438 MiB frame the run never takes.
    root = Path(checkout) / "woof"
    for path in root.rglob("*.py"):
        try:
            if kernel in path.read_text(encoding="utf-8", errors="ignore"):
                return True
        except Exception:
            continue
    return False


def first_zero_local_kernel(checkout, mod_path):
    """A kernel in this module with local_size_bytes == 0, for the negative
    control.  Chosen from the source so the control is a real kernel from the
    same compile, not a synthetic one."""
    cmd = [sys.executable, "-c", (
        "import sys,json;sys.path.insert(0,%r);sys.path.insert(0,%r);"
        "import cupy as cp;"
        "from reservation_probe import load_probe_module;"
        "from kernel_reservation import func_attributes;"
        "import re;"
        "o,src=load_probe_module(%r);"
        "names=re.findall(r'extern\\s+\"C\"\\s+__global__\\s+void\\s+([A-Za-z_]\\w*)',src);"
        "out=None\n"
        "for n in names:\n"
        "    try:\n"
        "        k=o.get_function(n)\n"
        "    except Exception:\n"
        "        continue\n"
        "    if (func_attributes(cp,k).get('local_size_bytes') or 0)==0:\n"
        "        out=n;break\n"
        "print(json.dumps(out))"
    ) % (str(HERE), checkout, mod_path)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return None


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if mode == "one":
        raise SystemExit(one(sys.argv[2:]))
    raise SystemExit(sweep(sys.argv[2:] if mode == "sweep" else sys.argv[1:]))
