"""kim_tofd measurements: cubin identity of the unchanged units, frames, kernel timing.

    python measure.py OLD_KERNEL_DIR CASE_DIR OUT_JSON

* cubin identity: sfclay and sfclay_dx2d compiled from this tree and from
  OLD_KERNEL_DIR (dc3753350b's woof/core/kernels) with the production options
  of the current numerics mode; equal images mean kim_tofd = .false. (and the
  4.7.1 pin) runs exactly today's machine code.
* frames: kernel_local_size_bytes of sfclay_column in sfclay_tofd.
* timing: CUDA-event median of 200 launches on the captured real CONUS
  12 km columns (one 424 x 299 plane), sfclay (today, no drag) vs
  sfclay_tofd (kim_tofd on).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import cupy as cp

from woof import nvrtc_ptx_cache as compiler, wrf_exact
from woof.core import kernels
from woof.core.kernels import module_options, module_source, get_kernel, kernel_local_size_bytes

old_dir, case, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
mode = "strict" if wrf_exact.ENABLED else "fast"
rec = {"mode": mode, "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
       "nvrtc": ".".join(map(str, cp.cuda.nvrtc.getVersion()))}


def image(name, kdir=None):
    src = module_source(name) if kdir is None else module_source(name, kernel_dir=kdir)
    opts = module_options(name)
    if wrf_exact.ENABLED:
        opts = wrf_exact.effective_options(opts)
    img, _ = compiler.compile_using_nvrtc(src, opts, None, name + ".cu")
    img = img.encode() if isinstance(img, str) else img
    return hashlib.sha256(src.encode()).hexdigest(), hashlib.sha256(img).hexdigest(), len(img)


ident = {}
for name in ("sfclay", "sfclay_dx2d"):
    s_new, i_new, n_new = image(name)
    s_old, i_old, n_old = image(name, old_dir)
    ident[name] = {"source_sha_new": s_new, "source_sha_old": s_old,
                   "image_sha_new": i_new, "image_sha_old": i_old,
                   "image_bytes": n_new, "identical_image": i_new == i_old}
rec["cubin_identity"] = ident
fn = get_kernel("sfclay_tofd", "sfclay_column")
rec["sfclay_tofd_local_bytes"] = int(kernel_local_size_bytes(fn))
rec["sfclay_local_bytes"] = int(kernel_local_size_bytes(get_kernel("sfclay", "sfclay_column")))

# timing on the real captured columns
sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_native import IN, read  # noqa: E402
from woof.core.physics_inventory import SFCLAY_OUTPUTS  # noqa: E402
from woof.core.sfclay import SFClayResult, launch_sfclay  # noqa: E402

path = sorted((case / "captures").glob("sfclayrev-15-*-in.bin"))[0]
head, factor, planes = read(path, len(IN))
nx, ny = int(head[0]), int(head[1])
f = {n: cp.ascontiguousarray(cp.asarray(planes[i])) for i, n in enumerate(IN)}


def fresh():
    res = {n: cp.zeros((ny, nx), np.float32) for n in SFCLAY_OUTPUTS}
    for n in ("znt", "ust", "ustm", "mol", "hfx", "qfx", "qsfc", "zol"):
        res[n][...] = f[n]
    return SFClayResult(**res)


def time_it(tofd):
    r = fresh()
    extra = {"varf": f["varf"], "tofd_factor": float(factor)} if tofd else {}
    args = (f["u"], f["v"], f["t"], f["qv"], f["p"], f["dz8w"], f["psfc"], f["tsk"],
            f["pblh"], f["mavail"], f["xland"], f["lakemask"], r)
    for _ in range(20):
        launch_sfclay(*args, option=1, dx=12000.0, **extra)
    # Kernel time: raw launches of the unit launch_sfclay picks, between one
    # event pair (the Python wrapper's validation is not the kernel).
    arrays = args[:12] + tuple(getattr(r, n) for n in SFCLAY_OUTPUTS)
    tail = (np.int32(1), np.int32(1), np.int32(0), np.int32(0), np.int32(nx * ny))
    if tofd:
        k = get_kernel("sfclay_tofd", "sfclay_column")
        kargs = arrays + (cp.full((ny, nx), np.float32(12000.0)), f["varf"], np.float32(factor)) + tail
    else:
        k = get_kernel("sfclay", "sfclay_column")
        kargs = arrays + (np.float32(12000.0),) + tail
    grid = ((nx * ny + 127) // 128,)
    reps = []
    for _ in range(5):
        a, b = cp.cuda.Event(), cp.cuda.Event()
        a.record()
        for _ in range(200):
            k(grid, (128,), kargs)
        b.record(); b.synchronize()
        reps.append(cp.cuda.get_elapsed_time(a, b) / 200.0)
    return float(np.median(reps))


off = time_it(False); on = time_it(True); off2 = time_it(False)
rec["kernel_ms_per_launch_median5x200"] = {"columns": nx * ny, "sfclay_today_no_drag": off,
                               "sfclay_today_no_drag_repeat": off2,
                               "sfclay_tofd_on": on, "ratio_on_over_off": on / ((off + off2) / 2)}
out.write_text(json.dumps(rec, indent=1) + "\n")
print(json.dumps(rec, indent=1))
