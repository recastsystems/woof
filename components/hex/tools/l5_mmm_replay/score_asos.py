#!/usr/bin/env python3
"""Score hex forecast arms against ASOS (T2, Td2, low ceiling, fog proxy), nearest cell, hourly leads.

Usage: score_asos.py ASOS.csv OUT.json LABEL=FORECAST_DIR [LABEL=FORECAST_DIR ...] [--core-km 250 --lat 37.6 --lon -122.6]
Low ceiling: lowest BKN/OVC/VV base <= 1500 ft (457 m) vs model cloud base (lowest level with qc >= 1e-5 kg/kg,
height of the layer bottom above ground) <= 457 m. Fog proxy: visibility <= 3 mi vs qc >= 1e-5 kg/kg in the lowest
model layer. Model cells are in the fine core only (great-circle distance from the point <= core km).
"""
import csv
import datetime as dt
import glob
import json
import math
import os
import sys

import numpy as np
import netCDF4

args = [a for a in sys.argv[1:]]
asos, out = args[0], args[1]
opt = lambda k, d: float(args[args.index(k) + 1]) if k in args else d
core_km, plat, plon = opt("--core-km", 250.0), opt("--lat", 37.6), opt("--lon", -122.6)
arms = [a.split("=", 1) for a in args[2:] if "=" in a and not a.startswith("--")]
START = dt.datetime(2024, 6, 21, 0)
QC_CLOUD, CEIL_M, VIS_MI = 1e-5, 457.2, 3.0


def gc_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


rows = list(csv.DictReader(open(asos)))
stations = {}
for r in rows:
    stations.setdefault(r["station"], (float(r["lat"]), float(r["lon"])))
stations = {s: ll for s, ll in stations.items() if gc_km(plat, plon, ll[0], ll[1]) <= core_km - 20}


def fnum(x):
    try:
        return float(x)
    except ValueError:
        return None


obs = {}
for r in rows:
    if r["station"] not in stations:
        continue
    t = dt.datetime.strptime(r["valid"], "%Y-%m-%d %H:%M")
    lead = round((t - START).total_seconds() / 3600)
    off = abs((t - (START + dt.timedelta(hours=lead))).total_seconds())
    if not 1 <= lead <= 18 or off > 15 * 60:
        continue
    key = (r["station"], lead)
    if key in obs and obs[key]["off"] <= off:
        continue
    ceil = None
    layers = [(r[f"skyc{i}"], fnum(r[f"skyl{i}"])) for i in (1, 2, 3)]
    for cover, base in layers:
        if cover in ("BKN", "OVC", "VV") and base is not None:
            ceil = base * 0.3048 if ceil is None else min(ceil, base * 0.3048)
    clear = any(c in ("CLR", "SKC", "FEW", "SCT", "BKN", "OVC", "VV") for c, _ in layers)
    tf, df, vs = fnum(r["tmpf"]), fnum(r["dwpf"]), fnum(r["vsby"])
    obs[key] = dict(off=off, t2=None if tf is None else (tf - 32) * 5 / 9 + 273.15,
                    td2=None if df is None else (df - 32) * 5 / 9 + 273.15,
                    low=None if not clear else (ceil is not None and ceil <= CEIL_M),
                    fog=None if vs is None else vs <= VIS_MI)


def model_fields(fdir, lead):
    t = START + dt.timedelta(hours=lead)
    d = netCDF4.Dataset(os.path.join(fdir, f"history.{t:%Y-%m-%d_%H}.00.00.nc"))
    lat, lon = np.degrees(d["latCell"][:]), np.degrees(d["lonCell"][:])
    lon = np.where(lon > 180, lon - 360, lon)
    qc = d["qc"][0]  # (nCells, nVertLevels)
    zg = d["zgrid"][:]  # (nCells, nVertLevelsP1)
    agl = zg[:, :-1] - zg[:, :1]
    cloudy = qc >= QC_CLOUD
    first = np.where(cloudy.any(axis=1), cloudy.argmax(axis=1), -1)
    base = np.where(first >= 0, agl[np.arange(len(first)), np.maximum(first, 0)], np.inf)
    q2, ps = d["q2"][0], d["surface_pressure"][0] / 100.0
    e = np.maximum(q2 * ps / (0.622 + 0.378 * q2), 1e-6)
    lne = np.log(e / 6.112)
    td2 = 243.5 * lne / (17.67 - lne) + 273.15
    return dict(lat=lat, lon=lon, t2=d["t2m"][0], td2=td2, low=base <= CEIL_M, fog=cloudy[:, 0])


result = dict(asos=asos, stations=sorted(stations), core_km=core_km, ceiling_m=CEIL_M, qc_cloud=QC_CLOUD,
              visibility_mi=VIS_MI, arms={})
for label, fdir in arms:
    pairs = {k: [] for k in ("t2", "td2", "low", "fog")}
    hourly = {}
    index = None
    for lead in range(1, 19):
        m = model_fields(fdir, lead)
        if index is None:
            index = {}
            for s, (la, lo) in stations.items():
                index[s] = int(np.argmin(gc_km(la, lo, m["lat"], m["lon"])))
        h = {k: [] for k in pairs}
        for s in stations:
            o = obs.get((s, lead))
            if not o:
                continue
            c = index[s]
            for k in pairs:
                if o[k] is None:
                    continue
                mv = m[k][c]
                item = (float(o[k]), float(mv)) if k in ("t2", "td2") else (bool(o[k]), bool(mv))
                pairs[k].append(item)
                h[k].append(item)
        hourly[lead] = h
    scores = {}
    for k in ("t2", "td2"):
        a = np.array(pairs[k])
        err = a[:, 1] - a[:, 0]
        scores[k] = dict(n=len(a), bias=round(float(err.mean()), 3), mae=round(float(np.abs(err).mean()), 3),
                         rmse=round(float(np.sqrt((err ** 2).mean())), 3))
    for k in ("low", "fog"):
        a = np.array(pairs[k], bool)
        o, f = a[:, 0], a[:, 1]
        hit, miss, fa, cn = int((o & f).sum()), int((o & ~f).sum()), int((~o & f).sum()), int((~o & ~f).sum())
        scores[k] = dict(n=len(a), hits=hit, misses=miss, false_alarms=fa, correct_negatives=cn,
                         pod=round(hit / max(hit + miss, 1), 3), far=round(fa / max(hit + fa, 1), 3),
                         csi=round(hit / max(hit + miss + fa, 1), 3), freq_bias=round((hit + fa) / max(hit + miss, 1), 3))
    series = {}
    for lead, h in hourly.items():
        row = {}
        for k in ("t2", "td2"):
            if h[k]:
                e = np.array(h[k])
                row[k + "_bias"] = round(float((e[:, 1] - e[:, 0]).mean()), 3)
        for k in ("low", "fog"):
            if h[k]:
                e = np.array(h[k], bool)
                row[k + "_obs_frac"] = round(float(e[:, 0].mean()), 3)
                row[k + "_model_frac"] = round(float(e[:, 1].mean()), 3)
                hit = int((e[:, 0] & e[:, 1]).sum())
                row[k + "_csi"] = round(hit / max(int((e[:, 0] | e[:, 1]).sum()), 1), 3)
        series[lead] = row
    result["arms"][label] = dict(forecast=fdir, scores=scores, series=series)
    print(label, json.dumps(scores))
json.dump(result, open(out, "w"), indent=1)
