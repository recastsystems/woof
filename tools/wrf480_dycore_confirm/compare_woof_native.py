"""Word-for-word: WOOF-ArWen per-step wrfout frames vs one native multi-frame wrfout."""
import json, re, sys
from datetime import datetime
from pathlib import Path
import numpy as np
from netCDF4 import Dataset

native, woofdir, out, step_s = sys.argv[1], Path(sys.argv[2]), sys.argv[3], float(sys.argv[4])
N = Dataset(native); N.set_auto_maskandscale(False)
times = ["".join(t.astype(str)) for t in N.variables["Times"][:]]
t0 = datetime.strptime(times[0], "%Y-%m-%d_%H:%M:%S")
pat = re.compile(r"wrfout_d01_(\d{4}-\d{2}-\d{2})_(\d{2})[_:](\d{2})[_:](\d{2})$")
frames = {}
for p in sorted(woofdir.rglob("wrfout_d01_*")):
    m = pat.match(p.name)
    if m:
        v = datetime.strptime(f"{m[1]} {m[2]}:{m[3]}:{m[4]}", "%Y-%m-%d %H:%M:%S")
        frames[int(round((v - t0).total_seconds() / step_s))] = p
res = dict(native=native, woof_frames={k: str(v) for k, v in frames.items()}, per_step={}, fields={})
tot_w = tot_d = 0
fields = {}
for step, path in sorted(frames.items()):
    W = Dataset(path); W.set_auto_maskandscale(False)
    nidx = step  # the native file holds one frame per step, frame 0 = initial state
    sw = sd = 0
    for name, var in N.variables.items():
        if var.dimensions[:1] != ("Time",) or var.dtype.kind == "S":
            continue
        rec = fields.setdefault(name, dict(words=0, different=0, steps_diff=[], status="ok"))
        if name not in W.variables:
            rec["status"] = "absent_in_woof"; continue
        x = np.asarray(var[nidx]); y = np.asarray(W.variables[name][:])
        if y.ndim == x.ndim + 1 and y.shape[0] == 1:
            y = y[0]
        if x.shape != y.shape:
            rec["status"] = f"shape native{x.shape} woof{y.shape}"; continue
        if x.dtype != y.dtype:
            rec["status"] = f"dtype native {x.dtype} woof {y.dtype}"; continue
        xv = np.ascontiguousarray(x).view(np.uint8).reshape(-1, x.dtype.itemsize)
        yv = np.ascontiguousarray(y).view(np.uint8).reshape(-1, y.dtype.itemsize)
        d = int((xv != yv).any(axis=1).sum())
        rec["words"] += x.size; rec["different"] += d
        if d:
            rec["steps_diff"].append(step)
            if "first" not in rec:
                bad = np.argwhere(x != y)
                if bad.size:
                    i = tuple(int(v) for v in bad[0])
                    rec["first"] = dict(step=step, index=i, native=float(x[i]), woof=float(y[i]))
        sw += x.size; sd += d
    res["per_step"][step] = dict(words=sw, different=sd)
    tot_w += sw; tot_d += sd
    W.close()
res["fields"] = fields
res["words"] = tot_w; res["different"] = tot_d
order = ["U", "V", "W", "PH", "T", "MU", "P", "QVAPOR", "QCLOUD", "QRAIN"]
res["prognostic_summary"] = {k: {kk: fields[k][kk] for kk in ("words", "different", "status")} for k in order if k in fields}
open(out, "w").write(json.dumps(res, indent=1) + "\n")
print(json.dumps(dict(frames=sorted(frames), words=tot_w, different=tot_d, per_step=res["per_step"])))
print("prognostic:", json.dumps(res["prognostic_summary"]))
bad = {k: v for k, v in fields.items() if v["different"] or v["status"] != "ok"}
print("not-ok:", json.dumps({k: (v["status"], v["different"], v.get("first")) for k, v in bad.items()})[:4000])
