"""WRF passes surface_driver call-locals (QGH, CHS, CQS, CPM, CHS2, CQS2,
module_surface_driver.F:1284-1290) to mynnsfc_driver uninitialised.  Show the
GPU outputs do not depend on their incoming words."""
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
import replay as R
tot = diff = 0
for p in sys.argv[2:]:
    cap = dict(np.load(p))
    a = R.run_gpu(cap)
    rng = np.random.default_rng(5)
    for k in ("qgh", "chs", "cqs", "cpm", "chs2", "cqs2"):
        cap[k] = rng.uniform(-1e3, 1e3, cap[k].shape).astype(np.float32)
    b = R.run_gpu(cap)
    names = R.COMPARED_WRAPPER if int(cap["mode"]) == 1 else R.COMPARED_SINGLE
    for n in names:
        x = np.asarray(a[n], np.float32).view(np.uint32); y = np.asarray(b[n], np.float32).view(np.uint32)
        tot += x.size; diff += int((x != y).sum())
print("locals-independence", tot, "words", diff, "different")
