import sys, glob, os, netCDF4, numpy as np
nat = netCDF4.Dataset(sys.argv[1]); nat.set_auto_maskandscale(False)
times = ["".join(t.astype(str)) for t in nat.variables["Times"][:]]
t = sys.argv[3]; k = times.index(t)
a = netCDF4.Dataset(glob.glob(os.path.join(sys.argv[2], "**", "wrfout_d01_" + t.replace(":", "?")), recursive=True)[0]); a.set_auto_maskandscale(False)
for v in [w for w in sys.argv[4].split(",") if w in a.variables and w in nat.variables]:
    x = np.asarray(a.variables[v][0]); y = np.asarray(nat.variables[v][k])
    d = x != y
    if not d.any(): print(v, "same"); continue
    xi = x.view(np.int32).astype(np.int64); yi = y.view(np.int32).astype(np.int64)
    idx = np.argwhere(d)
    print(v, "ndiff", int(d.sum()), "maxulp", int(np.abs(xi - yi)[d].max()), "first", idx[:3].tolist(), "vals", x[d][:3], y[d][:3])
