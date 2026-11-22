#!/usr/bin/env python3
"""WRF 4.8.0 Shin-Hong column replay: write inputs, read outputs, compare words.

    python -m tools.shinhong480_oracle.columns corpus IN.bin [MANIFEST.json]
    python -m tools.shinhong480_oracle.columns compare IN.bin WRF-OUT.bin REPORT.json
    python -m tools.shinhong480_oracle.columns time IN.bin REPORT.json

``corpus`` writes v2 records (the format run_shinhong480_columns.F90 reads)
from the wide-column regime builders of tools/shinhong_column_oracle
(129 base columns: 18 families and the edge probes), in four arms:

* ``default``: every base column at all six grid spacings (100 m to 27 km),
  WRF 4.8.0's Registry defaults (nonlocal flux on, scu mixing and KE
  dissipation off), topo_wind off;
* ``ysu-cg``: shinhong_nonlocal_flux = .false. (YSU counter-gradient heat
  flux), 3 km;
* ``scu-dissip``: shinhong_scu_mixing and shinhong_ke_dissipation on with a
  radiative cooling profile (stronger at cloud top), 3 km;
* ``topo``: topo_wind coefficients ctopo 0.85, ctopo2 0.7, 3 km.

Captured engine columns (``capture.py``) use the same record layout.

``compare`` runs woof.core.shinhong.launch_shinhong_480 on every record and
compares each output word with the native WRF 4.8.0 objects' (bit pattern;
NaN counted by geometry).  ``time`` measures the 4.8.0 and 4.6.1 kernels on
the same columns.  Nothing here knows a tolerance.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

F32 = np.float32
SCALARS = ("dt", "dx", "dy", "psfc", "znt", "ust", "hfx", "qfx", "wspd",
           "br", "psim", "psih", "xland", "u10", "v10", "corf", "ctopo",
           "ctopo2", "uoce", "voce")
LEVELS = ("u", "v", "theta", "pi", "t", "qv", "qc", "qi", "p", "dz", "tke",
          "rthraten")
WRF_LEVELS = ("du", "dv", "dtheta", "dqv", "dqc", "dqi", "exch_h", "exch_m",
              "tke", "el")
WRF_SURFACE = ("hpbl", "wstar", "delta", "u10", "v10")
ARMS = {"default": 1, "ysu-cg": 0, "scu-dissip": 7, "topo": 1}


def _v1_records():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.shinhong_column_oracle import make_columns as mc
    return mc.records()


def _cooling(rec):
    """A radiative theta tendency: clear-sky cooling, strong at cloud top."""
    nz = rec["nz"]
    r = np.full(nz, -2.0e-5, F32)                # about -1.7 K/day
    q = np.asarray(rec["qc"], F32) + np.asarray(rec["qi"], F32)
    cloudy = np.nonzero(q > 1.0e-5)[0]
    if cloudy.size:
        top = int(cloudy.max())
        r[top] = F32(-1.5e-3)                    # about -130 K/day at the top
        if top > 0:
            r[top - 1] = F32(-3.0e-4)
    return r


def corpus():
    recs, labels = _v1_records()
    out = []
    for rec, lab in zip(recs, labels):
        if rec["tke_diag"] != 1:                 # 4.8.0's chain always runs
            continue
        three_km = abs(float(lab["dx"]) - 3000.0) < 1.0
        for arm, flags in ARMS.items():
            if arm != "default" and not three_km:
                continue
            r = {k: (np.array(v, F32) if isinstance(v, np.ndarray) else v)
                 for k, v in rec.items() if k != "scalars"}
            s = dict(rec["scalars"])
            s["ctopo"], s["ctopo2"] = ((F32(0.85), F32(0.7)) if arm == "topo"
                                       else (F32(1.0), F32(1.0)))
            s["uoce"] = s["voce"] = F32(0.0)
            r["scalars"] = s
            r["rthraten"] = (_cooling(rec) if arm == "scu-dissip"
                             else np.zeros(rec["nz"], F32))
            r["flags"] = flags
            r["topo"] = 1 if arm == "topo" else 0
            r["arm"] = arm
            r["label"] = lab["label"]
            out.append(r)
    return out


def write_v2(recs, path):
    with open(path, "wb") as fh:
        for i, r in enumerate(recs):
            fh.write(struct.pack("<8i", r["nz"], r["flags"], r["family"],
                                 r["base"], r["topo"], i, 0, 0))
            fh.write(np.asarray([r["scalars"][k] for k in SCALARS],
                                "<f4").tobytes())
            for name in LEVELS:
                a = np.asarray(r[name], "<f4")
                assert a.shape == (r["nz"],), (name, a.shape)
                fh.write(a.tobytes())
            pin = np.asarray(r["p_interface"], "<f4")
            assert pin.shape == (r["nz"] + 1,)
            fh.write(pin.tobytes())


def _bytes(path):
    path = Path(path)
    if path.suffix == ".gz":
        import gzip
        return gzip.decompress(path.read_bytes())
    return path.read_bytes()


def read_v2(path):
    b = _bytes(path)
    o = 0
    recs = []
    while o < len(b):
        h = struct.unpack_from("<8i", b, o); o += 32
        nz = h[0]
        sc = np.frombuffer(b, "<f4", len(SCALARS), o); o += 4 * len(SCALARS)
        r = {"nz": nz, "flags": h[1], "family": h[2], "base": h[3],
             "topo": h[4], "case": h[5],
             "scalars": dict(zip(SCALARS, sc.astype(F32)))}
        for name in LEVELS:
            r[name] = np.frombuffer(b, "<f4", nz, o).astype(F32); o += 4 * nz
        r["p_interface"] = np.frombuffer(b, "<f4", nz + 1, o).astype(F32)
        o += 4 * (nz + 1)
        recs.append(r)
    return recs


def read_wrf_out(path, recs):
    b = _bytes(path)
    o = 0
    outs = []
    for r in recs:
        nz, kpbl, _, _ = struct.unpack_from("<4i", b, o); o += 16
        assert nz == r["nz"]
        d = {"kpbl": np.int32(kpbl)}
        for name in WRF_LEVELS:
            d[name] = np.frombuffer(b, "<f4", nz, o).astype(F32); o += 4 * nz
        sf = np.frombuffer(b, "<f4", 5, o).astype(F32); o += 20
        d.update(zip(WRF_SURFACE, sf))
        outs.append(d)
    assert o == len(b)
    return outs


def write_out(outs, recs, path):
    """The port's outputs in the native driver's output record layout."""
    with open(path, "wb") as fh:
        for r, d in zip(recs, outs):
            fh.write(struct.pack("<4i", r["nz"], int(d["kpbl"]), r["family"],
                                 r["base"]))
            for name in WRF_LEVELS:            # the 4.6.1 unit has no exch_m
                fh.write(np.asarray(d.get(name, np.zeros(r["nz"])),
                                    "<f4").tobytes())
            fh.write(np.asarray([d.get(n, 0.0) for n in WRF_SURFACE],
                                "<f4").tobytes())


def _groups(recs):
    g = {}
    for i, r in enumerate(recs):
        s = r["scalars"]
        key = (r["nz"], float(s["dt"]), float(s["dx"]), r["flags"], r["topo"])
        g.setdefault(key, []).append(i)
    return g


def run_port(recs, *, generation="wrf_480"):
    import cupy as cp
    if generation == "wrf_480":
        from woof.core.shinhong import launch_shinhong_480
    else:
        from woof.core.shinhong import launch_shinhong

    res = [None] * len(recs)
    for (nz, dt, dx, flags, topo), idx in sorted(_groups(recs).items()):
        n = len(idx)

        def lev(name):
            a = np.stack([recs[i][name] for i in idx], axis=1)
            return cp.asarray(np.ascontiguousarray(a.reshape(nz, 1, n)))

        def sfc(name):
            a = np.asarray([recs[i]["scalars"][name] for i in idx], F32)
            return cp.asarray(np.ascontiguousarray(a.reshape(1, n)))

        pin = np.stack([recs[i]["p_interface"] for i in idx], axis=1)
        pin = cp.asarray(np.ascontiguousarray(pin.reshape(nz + 1, 1, n)))
        common = dict(psfc=sfc("psfc"), znt=sfc("znt"), ust=sfc("ust"),
                      hfx=sfc("hfx"), qfx=sfc("qfx"), wspd=sfc("wspd"),
                      br=sfc("br"), psim=sfc("psim"), psih=sfc("psih"),
                      xland=sfc("xland"), u10=sfc("u10"), v10=sfc("v10"),
                      corf=sfc("corf"), dt=dt)
        if generation == "wrf_480":
            out = launch_shinhong_480(
                lev("u"), lev("v"), lev("theta"), lev("qv"), lev("qc"),
                lev("qi"), lev("p"), pin, lev("pi"), lev("dz"), lev("tke"),
                rthraten=lev("rthraten"), dx=dx,
                nonlocal_flux=bool(flags & 1), scu_mixing=bool(flags & 2),
                ke_dissipation=bool(flags & 4),
                uoce=sfc("uoce"), voce=sfc("voce"),
                topo=(sfc("ctopo"), sfc("ctopo2")) if topo else None,
                **common)
        else:
            out = launch_shinhong(
                lev("u"), lev("v"), lev("theta"), lev("qv"), lev("qc"),
                lev("qi"), lev("p"), pin, lev("pi"), lev("dz"), lev("tke"),
                dx=dx, dy=dx, tke_diag=1, **common)
        host = {k: cp.asnumpy(v) for k, v in out.items()}
        for j, i in enumerate(idx):
            res[i] = {k: (v[:, 0, j] if v.ndim == 3 else v[0, j])
                      for k, v in host.items()}
    return res


def compare(recs, wrf, port):
    fields = {name: {"words": 0, "different": 0, "nan_geometry": 0,
                     "max_ulp": 0} for name in WRF_LEVELS + WRF_SURFACE}
    fields["kpbl"] = {"words": 0, "different": 0}
    per_case = []
    for i, (w, p) in enumerate(zip(wrf, port)):
        bad = {}
        for name in WRF_LEVELS + WRF_SURFACE:
            a = np.atleast_1d(np.asarray(w[name], F32))
            b = np.atleast_1d(np.asarray(p[name], F32))
            na, nb = np.isnan(a), np.isnan(b)
            same = (a.view(np.uint32) == b.view(np.uint32)) | (na & nb)
            f = fields[name]
            f["words"] += a.size
            d = int((~same).sum())
            f["different"] += d
            f["nan_geometry"] += int((na != nb).sum())
            if d:
                ok = ~(na | nb)
                ia = a.view(np.int32).astype(np.int64)
                ib = b.view(np.int32).astype(np.int64)
                ia = np.where(ia < 0, -(ia & 0x7FFFFFFF), ia)
                ib = np.where(ib < 0, -(ib & 0x7FFFFFFF), ib)
                if ok.any():
                    f["max_ulp"] = max(f["max_ulp"],
                                       int(np.abs(ia - ib)[ok].max()))
                bad[name] = d
        fields["kpbl"]["words"] += 1
        if int(w["kpbl"]) != int(p["kpbl"]):
            fields["kpbl"]["different"] += 1
            bad["kpbl"] = 1
        if bad:
            r = recs[i]
            per_case.append({"case": i, "nz": r["nz"], "flags": r["flags"],
                             "topo": r["topo"], "family": r["family"],
                             "base": r["base"],
                             "dx": float(r["scalars"]["dx"]), "fields": bad})
    total = sum(f["words"] for f in fields.values())
    diff = sum(f["different"] for f in fields.values())
    nan_cases = sorted({i for i, w in enumerate(wrf)
                        if any(np.isnan(np.atleast_1d(w[n])).any()
                               for n in WRF_LEVELS + WRF_SURFACE)})
    return {"columns": len(recs), "words": total, "different": diff,
            "fields": fields, "differing_cases": per_case,
            "wrf_nan_cases": nan_cases}


def _main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    cmd = argv[1]
    if cmd == "corpus":
        recs = corpus()
        write_v2(recs, argv[2])
        if len(argv) > 3:
            Path(argv[3]).write_text(json.dumps(
                [{"case": i, "arm": r["arm"], "label": r["label"],
                  "family": r["family"], "base": r["base"], "nz": r["nz"],
                  "dx": float(r["scalars"]["dx"]),
                  "dt": float(r["scalars"]["dt"])}
                 for i, r in enumerate(recs)], indent=0) + "\n")
        print(f"wrote {len(recs)} columns to {argv[2]}")
        return 0
    if cmd == "time":
        # Kernel time, 4.8.0 vs 4.6.1 unit, same columns, same card: the
        # largest same-nz group of the default arm tiled to NCOL columns.
        import time as _time
        import cupy as cp
        from woof.core.shinhong import launch_shinhong, launch_shinhong_480
        recs = [r for r in read_v2(argv[2]) if r["flags"] == 1 and not r["topo"]]
        ncol = int(argv[4]) if len(argv) > 4 else 160000
        by = {}
        for r in recs:
            key = (r["nz"], float(r["scalars"]["dx"]), float(r["scalars"]["dt"]))
            by.setdefault(key, []).append(r)
        (nz, _, _), grp = max(by.items(), key=lambda kv: (len(kv[1]), kv[0]))
        reps = (ncol + len(grp) - 1) // len(grp)
        pick = (grp * reps)[:ncol]

        def lev(name):
            a = np.stack([r[name] for r in pick], axis=1)
            return cp.asarray(np.ascontiguousarray(a.reshape(nz, 1, ncol)))

        def sfc(name):
            return cp.asarray(np.asarray([r["scalars"][name] for r in pick],
                                         F32).reshape(1, ncol))
        pin = cp.asarray(np.ascontiguousarray(np.stack(
            [r["p_interface"] for r in pick], axis=1).reshape(nz + 1, 1, ncol)))
        L = {n: lev(n) for n in ("u", "v", "theta", "qv", "qc", "qi", "p",
                                 "pi", "dz", "tke")}
        S = {n: sfc(n) for n in ("psfc", "znt", "ust", "hfx", "qfx", "wspd",
                                 "br", "psim", "psih", "xland", "u10", "v10",
                                 "corf")}
        dt = float(pick[0]["scalars"]["dt"]); dx = float(pick[0]["scalars"]["dx"])
        args = (L["u"], L["v"], L["theta"], L["qv"], L["qc"], L["qi"], L["p"],
                pin, L["pi"], L["dz"], L["tke"])
        runs = {
            "wrf_461": lambda: launch_shinhong(*args, dt=dt, dx=dx, dy=dx,
                                               tke_diag=1, **S),
            "wrf_480": lambda: launch_shinhong_480(*args, dt=dt, dx=dx, **S),
        }
        rep = {"ncol": ncol, "nz": nz, "dt": dt, "dx": dx, "ms": {}}
        for name, fn in runs.items():
            fn(); cp.cuda.Device().synchronize()
            times = []
            for _ in range(int(argv[5]) if len(argv) > 5 else 20):
                e0, e1 = cp.cuda.Event(), cp.cuda.Event()
                e0.record(); fn(); e1.record(); e1.synchronize()
                times.append(cp.cuda.get_elapsed_time(e0, e1))
            rep["ms"][name] = {"median": float(np.median(times)),
                               "min": float(np.min(times)),
                               "max": float(np.max(times))}
        rep["ratio_480_over_461"] = (rep["ms"]["wrf_480"]["median"]
                                     / rep["ms"]["wrf_461"]["median"])
        import os
        rep["GPUWM_WRF_EXACT"] = os.environ.get("GPUWM_WRF_EXACT", "")
        name = cp.cuda.runtime.getDeviceProperties(0)["name"]
        rep["device"] = name.decode() if isinstance(name, bytes) else name
        Path(argv[3]).write_text(json.dumps(rep, indent=1) + chr(10))
        print(json.dumps(rep))
        return 0
    if cmd == "arm461":
        # The 4.6.1 unit (the wrf_461 option) on every record, outputs in
        # the driver's layout: run once with the tip on PYTHONPATH and once
        # with the branch, then compare the two files with "same".
        recs = read_v2(argv[2])
        write_out(run_port(recs, generation="wrf_461"), recs, argv[3])
        return 0
    if cmd == "same":
        recs = read_v2(argv[2])
        a, b = read_wrf_out(argv[3], recs), read_wrf_out(argv[4], recs)
        rep = compare(recs, a, b)
        print(f"{argv[3]} vs {argv[4]}: {rep['different']} of {rep['words']} words differ")
        if len(argv) > 5:
            Path(argv[5]).write_text(json.dumps(rep, indent=1) + chr(10))
        return 0
    if cmd == "reflag":
        # IN.bin OUT.bin FLAGS: the same columns under another option word
        # (1 nonlocal flux, 2 scu mixing, 4 KE dissipation), so captured real
        # columns exercise every 4.8.0 option against the native objects.
        recs = read_v2(argv[2])
        for r in recs:
            r["flags"] = int(argv[4])
        write_v2(recs, argv[3])
        return 0
    if cmd == "reverse":
        # The undefined-word witness: the same records in reverse order,
        # so each call meets different leftover stack memory.
        recs = read_v2(argv[2])
        for r in recs:
            r["label"] = ""
        write_v2(list(reversed(recs)), argv[3])
        return 0
    if cmd == "witness":
        # IN.bin, OUT.bin (forward order), REV-OUT.bin (reverse order):
        # words of each column that differ between the two native runs.
        recs = read_v2(argv[2])
        fwd = read_wrf_out(argv[3], recs)
        rev = list(reversed(read_wrf_out(argv[4], list(reversed(recs)))))
        rep = compare(recs, fwd, rev)
        Path(argv[5]).write_text(json.dumps(rep, indent=1) + chr(10))
        print(f"native forward vs reverse: {rep['different']} of "
              f"{rep['words']} words differ")
        for name, f in rep["fields"].items():
            if f["different"]:
                print(f"  {name}: {f['different']} of {f['words']}")
        return 0
    if cmd == "compare":
        recs = read_v2(argv[2])
        wrf = read_wrf_out(argv[3], recs)
        port = run_port(recs)
        write_out(port, recs, Path(argv[4]).with_suffix(".port-out.bin"))
        rep = compare(recs, wrf, port)
        import os
        rep["GPUWM_WRF_EXACT"] = os.environ.get("GPUWM_WRF_EXACT", "")
        import cupy as cp
        name = cp.cuda.runtime.getDeviceProperties(0)["name"]
        rep["device"] = name.decode() if isinstance(name, bytes) else name
        Path(argv[4]).write_text(json.dumps(rep, indent=1) + "\n")
        print(f"columns {rep['columns']} words {rep['words']} "
              f"different {rep['different']} (WRF NaN cases "
              f"{len(rep['wrf_nan_cases'])})")
        for name, f in rep["fields"].items():
            if f["different"]:
                print(f"  {name}: {f['different']} of {f['words']}"
                      f" max_ulp {f.get('max_ulp')} nan_geo {f.get('nan_geometry')}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
