"""The WRF v4.6.1 SAS (cu_physics = 94) column oracle and its GPU replay.

The fixtures under ``tests/data/oracles/sas/`` are written by
``tools/sas_wrf461_oracle/build.sh``: a harness linked against the objects of
the BUILT WRF trees (strict and scalar, which agree byte for byte), run over
120 synthetic columns in ten families (deep ocean/land, capped, shallow only,
dry, two-level, long time step, high top, cold, subsiding).  Files, gzip'd,
little endian, one record per column:

* ``cases.bin``   CU_SAS's REAL inputs (tools/sas_wrf461_oracle/gen_cases.py)
* ``stage1.bin``  SASCNVN: its kind_phys inputs and outputs
* ``stage2.bin``  OLD_ARW_SHALCV: its inputs (post-SASCNVN) and outputs
* ``stage3.bin``  CU_SAS: RTHCUTEN RQVCUTEN RQCCUTEN RQICUTEN RAINCV PRATEC HBOT HTOP
* ``tables.json`` SHA-256 of each funcphys table section after GFUNCPHYS

:func:`grade` replays each stage on the GPU and counts words that differ;
``python -m woof.verify.sas_oracle`` prints the table, ``--plant`` also
compiles a copy of sas.cu with one bit of one constant flipped and requires
the replay to catch it.  Run with ``GPUWM_WRF_EXACT=1`` for the strict arm.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from woof.verify.wrf471_fixtures import require_fixture_dir

ORACLE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "sas"
CASE_ORDER = ("t3d", "qv", "qc", "qi", "u", "v", "w", "pcps", "pi", "rho",
              "dz8w", "p8w")
CATEGORIES = ("deep-ocean", "deep-land", "capped", "shallow-only", "dry",
              "two-level", "long-dt", "high-top", "cold", "subsiding")

#: The planted mutation: the low bit of c0 (module_cu_sas.F :2692, the
#: autoconversion rate).  One ULP of one constant in one stage.
PLANT_FROM = "const SD c0 = SL(.002), c1 = SL(.002), delta = fv;"
PLANT_TO = ("const SD c0 = SD(__longlong_as_double(__double_as_longlong("
            "(double).002f) ^ 1LL)), c1 = SL(.002), delta = fv;")


class _Reader:
    def __init__(self, data: bytes):
        self.data, self.off = data, 0

    def take(self, fmt):
        vals = struct.unpack_from(fmt, self.data, self.off)
        self.off += struct.calcsize(fmt)
        return vals

    def arr(self, dtype, n):
        a = np.frombuffer(self.data, dtype=dtype, count=n, offset=self.off).copy()
        self.off += a.nbytes
        return a


@lru_cache(maxsize=None)
def _blob(name: str, directory: str | None = None) -> bytes:
    d = Path(directory) if directory else require_fixture_dir(
        ORACLE_DIR, "SAS", source="WRF v4.6.1")
    path = d / name
    if path.exists():
        return path.read_bytes()
    return gzip.decompress((d / (name + ".gz")).read_bytes())


def load_cases(directory=None):
    r = _Reader(_blob("cases.bin", directory))
    (n,) = r.take("<i")
    cases = []
    for _ in range(n):
        nz, stepcu = r.take("<ii")
        dt, pgcon, xland = r.take("<fff")
        c = dict(nz=nz, stepcu=stepcu, dt=dt, pgcon=pgcon, xland=xland)
        for name in CASE_ORDER:
            c[name] = r.arr("<f4", nz + 1 if name in ("w", "p8w") else nz)
        (c["category"],) = r.take("<i")
        cases.append(c)
    return cases


def load_stage1(directory=None):
    r = _Reader(_blob("stage1.bin", directory))
    (n,) = r.take("<i")
    recs = []
    for _ in range(n):
        (nz,) = r.take("<i")
        delt, pgcon, massf, ps, slimsk = r.take("<5d")
        rec = dict(nz=nz, delt=delt, pgcon=pgcon, massf=massf, ps=ps, slimsk=slimsk)
        for name in ("del", "prsl", "phil", "ql1", "ql2", "q1", "t1", "u1", "v1", "dot"):
            rec["in_" + name] = r.arr("<f8", nz)
        for name in ("ql1", "ql2", "q1", "t1", "u1", "v1"):
            rec["out_" + name] = r.arr("<f8", nz)
        rec["cldwrk"], rec["rn"] = r.take("<2d")
        rec["kbot"], rec["ktop"], rec["kcnv"] = r.take("<3i")
        recs.append(rec)
    return recs


def load_stage2(directory=None):
    r = _Reader(_blob("stage2.bin", directory))
    (n,) = r.take("<i")
    recs = []
    for _ in range(n):
        nz, kcnv = r.take("<ii")
        delt, dpshc = r.take("<2d")
        rec = dict(nz=nz, kcnv=kcnv, delt=delt, dpshc=dpshc)
        rec["del"] = r.arr("<f8", nz)
        rec["prsi"] = r.arr("<f8", nz + 1)
        for name in ("prsl", "prslk", "q_in", "t_in"):
            rec[name] = r.arr("<f8", nz)
        rec["q_out"] = r.arr("<f8", nz)
        rec["t_out"] = r.arr("<f8", nz)
        recs.append(rec)
    return recs


STAGE3_LEVEL = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten")
STAGE3_SURFACE = ("raincv", "pratec", "hbot", "htop")


def load_stage3(directory=None, shalconv=1):
    r = _Reader(_blob("stage3.bin" if shalconv else "stage3_shaloff.bin", directory))
    (n,) = r.take("<i")
    recs = []
    for _ in range(n):
        (nz,) = r.take("<i")
        rec = {"nz": nz}
        for name in STAGE3_LEVEL:
            rec[name] = r.arr("<f4", nz)
        vals = r.take("<4f")
        for name, v in zip(STAGE3_SURFACE, vals):
            rec[name] = np.float32(v)
        recs.append(rec)
    return recs


def table_digests(directory=None) -> dict:
    return json.loads(_blob("tables.json", directory).decode("utf-8"))


# ---- comparison ----------------------------------------------------------------
def _ordered(a: np.ndarray) -> list[int]:
    bits = 64 if a.dtype.itemsize == 8 else 32
    ints = a.view(np.int64 if bits == 64 else np.int32).astype(object)
    top = 1 << (bits - 1)
    return [(-top - i if i < 0 else i) for i in ints]


class Tally:
    """Words compared, words that differ, the largest ULP distance."""

    def __init__(self):
        self.words = 0
        self.mismatch = 0
        self.max_ulp = 0

    def add(self, got, want):
        got = np.ascontiguousarray(got)
        want = np.ascontiguousarray(want, dtype=got.dtype)
        self.words += got.size
        if got.dtype.kind in "iu":
            diff = got != want
            self.mismatch += int(diff.sum())
            if diff.any():
                self.max_ulp = max(self.max_ulp,
                                   int(np.abs(got.astype(np.int64) - want.astype(np.int64)).max()))
            return
        bits_g = got.view(np.uint64 if got.dtype.itemsize == 8 else np.uint32)
        bits_w = want.view(bits_g.dtype)
        diff = bits_g != bits_w
        self.mismatch += int(diff.sum())
        if diff.any():
            og, ow = _ordered(got[diff]), _ordered(want[diff])
            self.max_ulp = max(self.max_ulp, max(abs(x - y) for x, y in zip(og, ow)))

    def row(self):
        return {"words": self.words, "mismatch": self.mismatch, "max_ulp": self.max_ulp}


def _pad(arrs, rows, dtype):
    out = np.zeros((rows, len(arrs)), dtype=dtype)
    for j, a in enumerate(arrs):
        out[: len(a), j] = a
    return out


def grade_tables(directory=None) -> dict:
    import cupy as cp
    from woof.core import sas
    tb = cp.asnumpy(sas.tables())
    want = table_digests(directory)
    res = {}
    for name, (off, n) in sas.TABLE_SECTIONS.items():
        got = hashlib.sha256(np.ascontiguousarray(tb[off:off + n], dtype="<f8").tobytes()).hexdigest()
        res[name] = {"words": n, "match": got == want[name]}
    return res


def grade_stage1(directory=None, mod=None) -> dict:
    import cupy as cp
    from woof.core import sas
    recs = load_stage1(directory)
    n = len(recs)
    nzmax = max(r["nz"] for r in recs)
    km = cp.asarray(np.array([r["nz"] for r in recs], dtype=np.int32))
    sc = lambda key: cp.asarray(np.array([r[key] for r in recs], dtype=np.float64))
    lv = {name: cp.asarray(_pad([r["in_" + name] for r in recs], nzmax, np.float64))
          for name in ("del", "prsl", "phil", "ql1", "ql2", "q1", "t1", "u1", "v1", "dot")}
    cldwrk = cp.zeros(n, dtype=cp.float64)
    rn = cp.zeros(n, dtype=cp.float64)
    kbot, ktop, kcnv = (cp.zeros(n, dtype=cp.int32) for _ in range(3))
    ws = cp.empty(sas.SAS_SLOTS * (nzmax + 1) * n, dtype=cp.float64)
    kern = (mod or sas.module()).get_function("sas_stage_deep")
    kern(((n + 127) // 128,), (128,),
         (sas.tables(), ws, np.int32(nzmax + 1), np.int32(n), km,
          sc("delt"), sc("pgcon"), sc("massf"), sc("ps"), sc("slimsk"),
          lv["del"], lv["prsl"], lv["phil"], lv["ql1"], lv["ql2"], lv["q1"],
          lv["t1"], lv["u1"], lv["v1"], lv["dot"], cldwrk, rn, kbot, ktop, kcnv))
    host = {k: cp.asnumpy(v) for k, v in lv.items()}
    tallies = {name: Tally() for name in
               ("ql1", "ql2", "q1", "t1", "u1", "v1", "cldwrk", "rn", "kbot", "ktop", "kcnv")}
    got_s = {"cldwrk": cp.asnumpy(cldwrk), "rn": cp.asnumpy(rn), "kbot": cp.asnumpy(kbot),
             "ktop": cp.asnumpy(ktop), "kcnv": cp.asnumpy(kcnv)}
    for j, r in enumerate(recs):
        nz = r["nz"]
        for name in ("ql1", "ql2", "q1", "t1", "u1", "v1"):
            tallies[name].add(host[name][:nz, j], r["out_" + name])
        for name in ("cldwrk", "rn"):
            tallies[name].add(got_s[name][j:j + 1], np.array([r[name]]))
        for name in ("kbot", "ktop", "kcnv"):
            tallies[name].add(got_s[name][j:j + 1], np.array([r[name]], dtype=np.int32))
    return {k: t.row() for k, t in tallies.items()}


def grade_stage2(directory=None, mod=None) -> dict:
    import cupy as cp
    from woof.core import sas
    recs = load_stage2(directory)
    n = len(recs)
    nzmax = max(r["nz"] for r in recs)
    km = cp.asarray(np.array([r["nz"] for r in recs], dtype=np.int32))
    kc = cp.asarray(np.array([r["kcnv"] for r in recs], dtype=np.int32))
    sc = lambda key: cp.asarray(np.array([r[key] for r in recs], dtype=np.float64))
    pad = lambda key, rows: cp.asarray(_pad([r[key] for r in recs], rows, np.float64))
    q = pad("q_in", nzmax)
    t = pad("t_in", nzmax)
    ws = cp.empty(sas.SAS_SLOTS * (nzmax + 1) * n, dtype=cp.float64)
    kern = (mod or sas.module()).get_function("sas_stage_shal")
    kern(((n + 127) // 128,), (128,),
         (sas.tables(), ws, np.int32(nzmax + 1), np.int32(n), km, kc,
          sc("delt"), sc("dpshc"), pad("del", nzmax), pad("prsi", nzmax + 1),
          pad("prsl", nzmax), pad("prslk", nzmax), q, t))
    qh, th = cp.asnumpy(q), cp.asnumpy(t)
    tq, tt = Tally(), Tally()
    for j, r in enumerate(recs):
        tq.add(qh[: r["nz"], j], r["q_out"])
        tt.add(th[: r["nz"], j], r["t_out"])
    return {"q1": tq.row(), "t1": tt.row()}


def grade_stage3(directory=None, mod=None, shalconv=1) -> dict:
    from woof.core import sas
    cases = load_cases(directory)
    recs = load_stage3(directory, shalconv=shalconv)
    if len(cases) != len(recs) or any(c["nz"] != r["nz"] for c,r in zip(cases,recs)):
        raise ValueError("SAS capture input/output column counts or levels differ")
    nzmax = max(c["nz"] for c in cases)
    lev = {name: _pad([c[name] for c in cases],
                      nzmax + 1 if name in ("w", "p8w") else nzmax, np.float32)
           for name in CASE_ORDER}
    import cupy as cp
    out, _ = sas.run_columns(
        **{("pi" if k == "pi" else k): lev[k] for k in CASE_ORDER},
        xland=np.array([c["xland"] for c in cases], dtype=np.float32),
        dt=0.0, stepcu=1, pgcon=0.0,
        kmcol=np.array([c["nz"] for c in cases], dtype=np.int32),
        dtcol=np.array([c["dt"] for c in cases], dtype=np.float32),
        stepcucol=np.array([c["stepcu"] for c in cases], dtype=np.int32),
        pgconcol=np.array([c["pgcon"] for c in cases], dtype=np.float32),
        mod=mod, shalconv=shalconv)
    host = {k: cp.asnumpy(v) for k, v in out.items()}
    tallies = {name: Tally() for name in STAGE3_LEVEL + STAGE3_SURFACE}
    for j, r in enumerate(recs):
        nz = r["nz"]
        for name in STAGE3_LEVEL:
            tallies[name].add(host[name][:nz, j], r[name])
        for name in STAGE3_SURFACE:
            tallies[name].add(host[name][j:j + 1], np.array([r[name]], dtype=np.float32))
    return {k: t.row() for k, t in tallies.items()}


def planted_module(stage="deep"):
    """sas.cu with PLANT_FROM's c0 one ULP off, compiled on the loader's route."""
    import cupy as cp
    from woof.core.kernels import module_source
    src = module_source("gfs_sas")
    before, after = PLANT_FROM, PLANT_TO
    if stage == "driver":
        before = "raincv[g] = __fadd_rn(raincv1, 0.0f);"
        after = "raincv[g] = __uint_as_float(__float_as_uint(__fadd_rn(raincv1, 0.0f)) ^ 1U);"
    elif stage != "deep":
        raise ValueError(f"unknown mutation stage {stage!r}")
    if src.count(before) != 1:
        raise RuntimeError("the planted-mutation anchor is not in sas.cu exactly once")
    src = src.replace(before, after)
    mod = cp.RawModule(code=src, options=("-std=c++17",))
    from woof.certify.kernel_manifest import record_module
    record_module(f"woof.verify.sas_oracle:planted-{stage}", source=src,
                  options=("-std=c++17",), module=mod)
    return mod


def grade(directory=None, plant=False) -> dict:
    from woof import wrf_exact
    report = {"strict": bool(wrf_exact.ENABLED),
              "tables": grade_tables(directory),
              "stage1_sascnvn": grade_stage1(directory),
              "stage2_old_arw_shalcv": grade_stage2(directory),
              "stage3_cu_sas": grade_stage3(directory),
              "stage3_shallow_off": grade_stage3(directory, shalconv=0)}
    if plant:
        mod = planted_module()
        report["planted"] = {"stage1_sascnvn": grade_stage1(directory, mod),
                             "stage3_cu_sas": grade_stage3(directory, planted_module("driver"))}
    return report


def _mismatches(stage: dict) -> int:
    return sum(v.get("mismatch", 0) for v in stage.values())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="grade the SAS GPU port against WRF v4.6.1")
    ap.add_argument("--dir", default=None, help="oracle directory (default: the fixtures)")
    ap.add_argument("--plant", action="store_true", help="also require a planted bit flip to be caught")
    ap.add_argument("--json", default=None)
    ap.add_argument("--real-capture", type=Path, help="actual CU_SAS entry/exit capture, stage 3 only")
    args = ap.parse_args(argv)
    if args.real_capture is not None:
        from woof.numerics import is_strict
        if not is_strict():
            ap.error("real-input bitwise proof requires GPUWM_NUMERICS=strict")
        if args.dir is not None:
            ap.error("--dir and --real-capture are separate input sources")
        rows = grade_stage3(str(args.real_capture))
        rep = {"strict":True,"source":"real-native-run","stage3_cu_sas":rows}
        ok = _mismatches(rows)==0
        if args.plant:
            planted=grade_stage3(str(args.real_capture),planted_module("driver"))
            rep["planted_driver"]=planted
            ok &= _mismatches(planted)>0
        if args.json:
            Path(args.json).write_text(json.dumps(rep,indent=2)+"\n")
        print(json.dumps(rep,indent=2))
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    rep = grade(args.dir, plant=args.plant)
    print(f"strict arithmetic (GPUWM_WRF_EXACT=1): {rep['strict']}")
    for name, row in rep["tables"].items():
        print(f"table {name:5s} {row['words']:7d} words  {'identical' if row['match'] else 'DIFFERS'}")
    ok = all(r["match"] for r in rep["tables"].values())
    for stage in ("stage1_sascnvn", "stage2_old_arw_shalcv", "stage3_cu_sas", "stage3_shallow_off"):
        for field, row in rep[stage].items():
            print(f"{stage:22s} {field:9s} words {row['words']:6d} mismatch {row['mismatch']:5d} max_ulp {row['max_ulp']}")
        ok &= _mismatches(rep[stage]) == 0
    if args.plant:
        caught = {k: _mismatches(v) for k, v in rep["planted"].items()}
        print(f"planted one-bit deep constant and driver output flips: mismatching words {caught}")
        ok &= all(v > 0 for v in caught.values())
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
