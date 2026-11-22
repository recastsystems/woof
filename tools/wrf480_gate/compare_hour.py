"""Whole-forecast word compare: WOOF-ArWen per-frame wrfout files vs one native
multi-frame wrfout (history every step). Raw-byte comparison per variable per
frame over variables both write. Usage: compare_hour.py NATIVE.nc ARWEN_DIR OUT.json"""
import glob, json, sys, os
import netCDF4, numpy as np
nat_path, arw_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
nat = netCDF4.Dataset(nat_path); nat.set_auto_maskandscale(False)
times = ["".join(t.astype(str)) for t in nat.variables["Times"][:]]
idx = {t: i for i, t in enumerate(times)}
files = sorted(glob.glob(os.path.join(arw_dir, "**", "wrfout_d01_*"), recursive=True))
res = {"native": nat_path, "arwen_dir": arw_dir, "frames": 0, "variables": set(),
       "words": 0, "different_words": 0, "per_var": {}, "only_native": set(), "only_arwen": set(),
       "first_diff": {}}
for f in files:
    a = netCDF4.Dataset(f); a.set_auto_maskandscale(False)
    at = "".join(a.variables["Times"][0].astype(str))
    if at not in idx:
        a.close(); continue
    k = idx[at]; res["frames"] += 1
    na, nb = set(a.variables), set(nat.variables)
    res["only_native"] |= nb - na; res["only_arwen"] |= na - nb
    for v in sorted(na & nb):
        if v == "Times": continue
        va, vb = a.variables[v], nat.variables[v]
        if "Time" not in vb.dimensions[:1]:
            continue
        x = np.asarray(va[0]); y = np.asarray(vb[k])
        pv = res["per_var"].setdefault(v, {"words": 0, "different": 0})
        if x.shape != y.shape or x.dtype != y.dtype:
            pv.setdefault("mismatch", f"{x.shape}{x.dtype} vs {y.shape}{y.dtype}"); continue
        n = x.size
        d = int(np.count_nonzero(x.view(np.uint8).reshape(n, -1) != y.view(np.uint8).reshape(n, -1), axis=0).max() if False else
                np.count_nonzero(np.any(x.reshape(n).view(np.uint8).reshape(n, -1) != y.reshape(n).view(np.uint8).reshape(n, -1), axis=1)))
        pv["words"] += n; pv["different"] += d
        res["words"] += n; res["different_words"] += d; res["variables"].add(v)
        if d and v not in res["first_diff"]:
            res["first_diff"][v] = at
    a.close()
res["native_frames"] = len(times)
res["variables"] = len(res["variables"])
res["only_native"] = sorted(res["only_native"]); res["only_arwen"] = sorted(res["only_arwen"])
res["differing"] = {v: p for v, p in res["per_var"].items() if p["different"] or "mismatch" in p}
del res["per_var"]
# A compare that matched no frame, or fewer frames than native wrote, compared nothing or only a part:
# it used to print "different_words": 0 and exit 0 (seen 2026-11-09, after both ArWen runs had exited 2).
res["refused"] = None
if res["frames"] == 0 or res["frames"] != res["native_frames"] or res["words"] == 0:
    res["refused"] = (f"REFUSED: {res['frames']} of {res['native_frames']} native frames matched a WOOF-ArWen file "
                      f"({len(files)} wrfout_d01_* files under {arw_dir}), {res['words']} words compared; "
                      "this is not a whole-forecast compare and proves nothing")
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("frames", "variables", "words", "different_words")}),
      "differing:", {v: (p.get("different"), res["first_diff"].get(v)) for v, p in res["differing"].items()})
if res["refused"]:
    print(res["refused"], file=sys.stderr); sys.exit(2)
