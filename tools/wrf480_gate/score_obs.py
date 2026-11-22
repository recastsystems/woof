"""Score WOOF-ArWen arms at +1 h against ASOS (T2, Td2, 10 m wind) and MRMS
(composite reflectivity, 1 h QPE). Usage: score_obs.py OUT.json ARM=DIR ..."""
import sys, glob, os, json, csv, datetime as dt
import numpy as np, netCDF4
from scipy.ndimage import uniform_filter
sys.path.insert(0, os.environ.get("WOOF_TREE", "."))
from woof.obs.obspack import read_pack
O = os.path.expanduser("~/wrf480-port/gate/obs")
VALID = dt.datetime(2024, 5, 21, 22, 0)
def frame(d, t):
    f = glob.glob(os.path.join(d, "**", "wrfout_d01_" + t.strftime("%Y-%m-%d_%H_%M_%S")), recursive=True)
    f = f or glob.glob(os.path.join(d, "**", "wrfout_d01_" + t.strftime("%Y-%m-%d_%H:%M:%S")), recursive=True)
    ds = netCDF4.Dataset(f[0]); ds.set_auto_maskandscale(False); return ds
def td_from_q(q, p):
    e = q * p / (0.622 + q) / 100.0
    a = np.log(np.maximum(e, 1e-6) / 6.112)
    return 243.5 * a / (17.67 - a) + 273.15
def fss(f, o, thr, n):
    bf, bo = (f >= thr).astype(float), (o >= thr).astype(float)
    pf, po = uniform_filter(bf, n, mode="constant"), uniform_filter(bo, n, mode="constant")
    num = np.mean((pf - po) ** 2); den = np.mean(pf ** 2) + np.mean(po ** 2)
    return float(1 - num / den) if den > 0 else float("nan")
# ASOS: report nearest 22:00 within 15 min per station
rep = {}
for r in csv.DictReader(open(os.path.join(O, "asos.csv"))):
    try:
        t = dt.datetime.strptime(r["valid"], "%Y-%m-%d %H:%M")
        if abs((t - VALID).total_seconds()) > 900 or r["tmpf"] in ("", "M"): continue
        k = r["station"]
        if k not in rep or abs((t - VALID).total_seconds()) < abs((rep[k][0] - VALID).total_seconds()):
            rep[k] = (t, r)
    except Exception: pass
mr = read_pack(os.path.join(O, "refl.pack")); mq = read_pack(os.path.join(O, "qpe.pack"))
def regrid(pack, lat, lon):
    g = pack.meta["grid"]; v = pack.array("values")
    # 0.01 degree lattice: source origin per MRMS CONUS (NW corner 54.995N, -129.995E)
    la0 = 54.995 - 0.01 * g["j_start"]; lo0 = -129.995 + 0.01 * g["i_start"]
    j = np.clip(np.rint((la0 - lat) / 0.01).astype(int), 0, v.shape[0] - 1)
    i = np.clip(np.rint((lon - lo0) / 0.01).astype(int), 0, v.shape[1] - 1)
    return v[j, i]
out = {"valid": VALID.isoformat(), "arms": {}}
for arg in sys.argv[2:]:
    name, d = arg.split("=", 1)
    a = frame(d, VALID); a0 = frame(d, VALID - dt.timedelta(hours=1))
    lat = np.asarray(a.variables["XLAT"][0]); lon = np.asarray(a.variables["XLONG"][0])
    t2 = np.asarray(a.variables["T2"][0]); q2 = np.asarray(a.variables["Q2"][0]); ps = np.asarray(a.variables["PSFC"][0])
    u, v = np.asarray(a.variables["U10"][0]), np.asarray(a.variables["V10"][0])
    td = td_from_q(q2, ps)
    st = {"T2": [], "Td2": [], "wspd": [], "n": 0, "stations": []}
    for k, (t, r) in rep.items():
        sl, so = float(r["lat"]), float(r["lon"])
        if not (lat.min() < sl < lat.max() and lon.min() < so < lon.max()): continue
        jj, ii = np.unravel_index(np.argmin((lat - sl) ** 2 + ((lon - so) * np.cos(np.radians(sl))) ** 2), lat.shape)
        if min(jj, ii, lat.shape[0] - 1 - jj, lat.shape[1] - 1 - ii) < 5: continue  # skip boundary zone
        ot = (float(r["tmpf"]) - 32) / 1.8 + 273.15
        st["T2"].append(float(t2[jj, ii] - ot))
        if r["dwpf"] not in ("", "M"): st["Td2"].append(float(td[jj, ii] - ((float(r["dwpf"]) - 32) / 1.8 + 273.15)))
        if r["sknt"] not in ("", "M"): st["wspd"].append(float(np.hypot(u[jj, ii], v[jj, ii]) - float(r["sknt"]) * 0.514444))
        st["stations"].append(k)
    st["n"] = len(st["stations"])
    s = {kk: {"bias": float(np.mean(vv)), "rmse": float(np.sqrt(np.mean(np.square(vv)))), "n": len(vv)}
         for kk, vv in st.items() if kk in ("T2", "Td2", "wspd") and vv}
    s["stations"] = sorted(st["stations"])
    # reflectivity and QPE inside the non-boundary interior
    refl_name = "REFL_10CM" if "REFL_10CM" in a.variables else None
    cref = np.asarray(a.variables[refl_name][0]).max(axis=0) if refl_name else np.asarray(a.variables["REFD_MAX"][0])
    qpe = (np.asarray(a.variables["RAINNC"][0]) - np.asarray(a0.variables["RAINNC"][0])
           + np.asarray(a.variables["RAINC"][0]) - np.asarray(a0.variables["RAINC"][0]))
    oref, oqpe = regrid(mr, lat, lon), regrid(mq, lat, lon)
    I = (slice(5, -5), slice(5, -5))
    s["refl"] = {"model_max": float(cref[I].max()), "obs_max": float(oref[I].max()),
                 "model_area_ge30": float((cref[I] >= 30).mean()), "obs_area_ge30": float((oref[I] >= 30).mean()),
                 "fss30_n5": fss(cref[I], oref[I], 30, 5), "fss40_n5": fss(cref[I], oref[I], 40, 5)}
    s["qpe"] = {"model_mean_mm": float(qpe[I].mean()), "obs_mean_mm": float(oqpe[I].mean()),
                "model_max": float(qpe[I].max()), "obs_max": float(oqpe[I].max()),
                "fss1mm_n5": fss(qpe[I], oqpe[I], 1.0, 5), "fss5mm_n5": fss(qpe[I], oqpe[I], 5.0, 5),
                "rmse_mm": float(np.sqrt(np.mean((qpe[I] - oqpe[I]) ** 2)))}
    s["refl_field"] = refl_name or "REFD_MAX"
    out["arms"][name] = s
json.dump(out, open(sys.argv[1], "w"), indent=1)
for n, s in out["arms"].items():
    print(n, {k: (round(v["bias"], 2), round(v["rmse"], 2), v["n"]) for k, v in s.items() if k in ("T2", "Td2", "wspd")},
          {k: round(v, 3) for k, v in s["refl"].items()}, {k: round(v, 3) for k, v in s["qpe"].items()})
