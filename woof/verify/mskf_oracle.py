"""Grade the MSKF (cu_physics = 11) port against its WRF v4.6.1 oracle.

The fixture is what ``tools/mskf_wrf461_oracle/build.sh`` writes: the case
inputs (``cases_i32.bin``, ``cases_f32.bin``, ``cases.json``) and WRF's
outputs (``out_f32.bin``), every word little-endian float32.  This module
replays each case through the port exactly as ``run_mskf.F90`` drives
MSKF_CPS -- the W0AVG update first, every output pre-filled with the
oracle's -7777.0 sentinel, the case's own NCA -- and compares every output
word bitwise.

Grading is ``max_ulp == 0`` on every word of every field, including the
sentinel words a held column or an untouched level must leave alone.
"""

from __future__ import annotations

import json
import os
import sys
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from woof.core.fp32_ulp import monotone_fp32_key

FIELDS_3D = ("rthcuten", "rqvcuten", "rqccuten", "rqrcuten", "rqicuten",
             "rqscuten", "rucuten", "rvcuten", "cldfra_dp", "cldfra_sh",
             "qc_cu", "qi_cu", "qr_cu", "qs_cu", "nc_cu", "ni_cu", "nr_cu",
             "ns_cu", "ccn_cu", "efcs", "efis", "efss", "udr_kf", "ddr_kf",
             "uer_kf", "der_kf", "w0avg")
FIELDS_2D = ("raincv", "pratec", "nca", "cutop", "cubot", "cu_act",
             "ainc_frac", "timec_kf")
SENTINEL = np.float32(-7777.0)
FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data"
               / "oracles" / "mskf")


@dataclass
class Fixture:
    meta: dict
    ints: np.ndarray        # (ncase, 9) int32
    recs: np.ndarray        # (ncase, record_floats) float32
    out: np.ndarray         # (ncase, 27*nz + 8) float32

    @property
    def nz(self) -> int:
        return int(self.meta["nz"])

    @property
    def ncase(self) -> int:
        return int(self.meta["ncase"])


def load_fixture(directory: Path = FIXTURE_DIR) -> Fixture:
    directory = Path(directory)
    meta = json.loads((directory / "cases.json").read_text())
    n, nz = meta["ncase"], meta["nz"]
    ints = np.fromfile(directory / "cases_i32.bin", "<i4").reshape(n, -1)
    recs = np.fromfile(directory / "cases_f32.bin", "<f4").reshape(n, -1)
    out = np.fromfile(directory / "out_f32.bin", "<f4").reshape(
        n, len(FIELDS_3D) * nz + len(FIELDS_2D))
    return Fixture(meta, ints, recs, out)


def case_columns(fx: Fixture, c: int) -> dict:
    """Case ``c``'s inputs as host float32 arrays (column shape (nz,1,1))."""
    nz = fx.nz
    rec = fx.recs[c]
    nscal = len(fx.meta["scalars"])
    names = fx.meta["level_fields"]
    cols = {}
    for i, name in enumerate(names):
        cols[name] = rec[nscal + i * nz: nscal + (i + 1) * nz].copy()
    base = nscal + len(names) * nz
    cols["w"] = rec[base: base + nz + 1].copy()
    base += nz + 1
    cols["aerocu"] = rec[base: base + 10 * nz].reshape(10, nz).copy()
    scal = dict(zip(fx.meta["scalars"], (float(x) for x in rec[:nscal])))
    scal_f32 = dict(zip(fx.meta["scalars"], rec[:nscal]))
    ints = dict(zip(fx.meta["ints"], (int(x) for x in fx.ints[c])))
    return {"cols": cols, "scalars": scal, "scalars_f32": scal_f32,
            "ints": ints}


def run_port_case(fx: Fixture, c: int, *, kernel_override=None) -> np.ndarray:
    """One case through the port; the oracle's word layout comes back."""
    import cupy as cp

    from woof.core import mskf

    nz = fx.nz
    case = case_columns(fx, c)
    cols, sc, ci = case["cols"], case["scalars"], case["ints"]

    def dev3(a):
        return cp.asarray(np.ascontiguousarray(a, dtype=np.float32)
                          .reshape(nz, 1, 1))

    def dev2(x):
        return cp.full((1, 1), np.float32(x), dtype=cp.float32)

    inputs = {n: dev3(cols[n]) for n in
              ("u", "v", "t", "qv", "p", "rho", "pi", "dz8w", "w0avg")}
    for n in ("hfx", "ust", "pblh", "zol"):
        inputs[n] = dev2(case["scalars_f32"][n])
    w = cp.asarray(cols["w"].reshape(nz + 1, 1, 1))
    mskf.update_w0avg(inputs["w0avg"], w, dt=sc["dt"],
                      cudt_minutes=sc["cudt"], stepcu=ci["stepcu"],
                      adaptive=bool(ci["adapt_step"]))
    outputs = {n: cp.full((nz, 1, 1), SENTINEL, dtype=cp.float32)
               for n in mskf.OUTPUT_3D}
    for n in mskf.OUTPUT_2D:
        outputs[n] = cp.full((1, 1), SENTINEL, dtype=cp.float32)
    outputs["nca"][...] = np.float32(sc["nca"])
    outputs["cu_act"] = cp.zeros((1, 1), dtype=cp.int32)
    phase = mskf.MskfPhase(bool(ci["warm_rain"]), bool(ci["f_qr"]),
                           bool(ci["f_qi"]), bool(ci["f_qs"]))
    aerocu = (cp.asarray(cols["aerocu"].reshape(10, nz, 1, 1))
              if ci["aercu_opt"] > 0 else None)
    launch = kernel_override or mskf.launch_mskf
    status = launch(inputs, outputs, dx=sc["dx"], dt=sc["dt"],
                    cudt_minutes=sc["cudt"], phase=phase,
                    aercu_opt=ci["aercu_opt"], aercu_fct=sc["aercu_fct"],
                    kf_edrates=ci["kf_edrates"], aerocu=aerocu)
    words = []
    names3 = {"qc_cu": "qc_cu", "qi_cu": "qi_cu"}
    for name in FIELDS_3D:
        if name == "w0avg":
            words.append(cp.asnumpy(inputs["w0avg"]).reshape(nz))
        else:
            words.append(cp.asnumpy(outputs[names3.get(name, name)])
                         .reshape(nz))
    for name in FIELDS_2D:
        if name == "cu_act":
            words.append(np.array([float(cp.asnumpy(outputs["cu_act"])
                                         .reshape(-1)[0])], np.float32))
        else:
            words.append(cp.asnumpy(outputs[name]).reshape(1))
    out = np.concatenate(words).astype(np.float32)
    mskf.raise_on_status(status)
    return out


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|ULP| distance between float32 words, ordered-integer form."""
    return np.abs(monotone_fp32_key(a) - monotone_fp32_key(b))


@dataclass
class Grade:
    words: int
    differing: int
    max_ulp: int
    per_field: dict
    per_case: dict
    per_field_error: dict


def grade(fx: Fixture, port: np.ndarray, cases=None) -> Grade:
    cases = list(range(fx.ncase)) if cases is None else list(cases)
    nz = fx.nz
    ref = fx.out[cases]
    got = port
    diff = ref.view(np.uint32) != got.view(np.uint32)
    ulp = ulp_distance(ref, got)
    per_field = {}
    per_field_error = {}
    def errors(a, b):
        active = a != SENTINEL
        delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
        scale = float(np.max(np.abs(a[active]))) if active.any() else 0.0
        max_abs = float(np.max(delta[active])) if active.any() else 0.0
        return {"max_abs": max_abs, "reference_max_abs": scale,
                "normalized_max_abs": max_abs / scale if scale else (0.0 if max_abs == 0 else float("inf")),
                "sentinel_differences": int(np.count_nonzero((~active) & (a.view(np.uint32) != b.view(np.uint32)))),
                "reference_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                "port_sha256": hashlib.sha256(b.tobytes()).hexdigest()}
    for i, name in enumerate(FIELDS_3D):
        sl = slice(i * nz, (i + 1) * nz)
        per_field[name] = (int(diff[:, sl].sum()), int(ulp[:, sl].max()))
        per_field_error[name] = errors(ref[:, sl], got[:, sl])
    base = len(FIELDS_3D) * nz
    for i, name in enumerate(FIELDS_2D):
        per_field[name] = (int(diff[:, base + i].sum()),
                           int(ulp[:, base + i].max()))
        per_field_error[name] = errors(ref[:, base+i], got[:, base+i])
    per_case = {cases[j] + 1: int(diff[j].sum())
                for j in range(len(cases)) if diff[j].any()}
    return Grade(int(diff.size), int(diff.sum()), int(ulp.max()),
                 per_field, per_case, per_field_error)


def run_port(fx: Fixture, cases=None, **kw) -> np.ndarray:
    cases = list(range(fx.ncase)) if cases is None else list(cases)
    return np.stack([run_port_case(fx, c, **kw) for c in cases])


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", type=Path, default=FIXTURE_DIR)
    ap.add_argument("--cases", default="")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--rtol", type=float, default=0.001,
                    help="Default arm: allowed max error / reference field peak (sentinels excluded)")
    ap.add_argument("--atol", type=float, default=1e-10)

    args = ap.parse_args(argv)
    fx = load_fixture(args.fixture)
    cases = ([int(x) - 1 for x in args.cases.split(",") if x]
             or list(range(fx.ncase)))
    if not all(0 <= c < fx.ncase for c in cases):
        ap.error("case numbers must be in 1..ncase")
    if args.rtol < 0 or args.atol < 0:
        ap.error("tolerances must be nonnegative")
    values, valid, failures = [], [], {}
    for c in cases:
        try:
            value = run_port_case(fx, c)
            if not np.isfinite(value).all():
                raise FloatingPointError("MSKF returned a non-finite output word")
            values.append(value)
            valid.append(c)
        except (NotImplementedError, ValueError, FloatingPointError) as exc:
            failures[c + 1] = {"type": type(exc).__name__, "error": str(exc)}
    from woof.certify.kernel_manifest import kernel_manifest
    receipt = {"requested_cases": [c+1 for c in cases], "checked_cases": [c+1 for c in valid],
               "errors": failures, "strict": os.environ.get("GPUWM_WRF_EXACT") == "1",
               "command": [sys.executable, "-m", "woof.verify.mskf_oracle"] + sys.argv[1:],
               "tolerance": {"rtol": args.rtol, "atol": args.atol,
                             "rule": "max absolute error <= atol + rtol * reference field peak; sentinels must match bits"},
               "kernel_manifest": kernel_manifest()}
    passed = False
    if valid:
        g = grade(fx, np.stack(values), valid)
        receipt.update(g.__dict__)
        within = all(e["max_abs"] <= args.atol + args.rtol*e["reference_max_abs"]
                     and e["sentinel_differences"] == 0 for e in g.per_field_error.values())
        receipt["within_tolerance"] = within
        print(f"words {g.words}, differing {g.differing}, max_ulp {g.max_ulp}, within_tolerance {within}")
        for name, (n, m) in g.per_field.items():
            if n:
                e = g.per_field_error[name]
                print(f"  {name:10s} {n:6d} words, max_ulp {m}, max_abs {e['max_abs']:.9g}, normalized {e['normalized_max_abs']:.9g}")
        passed = not failures and (g.differing == 0 if receipt["strict"] else within)
    receipt["passed"] = passed
    if failures:
        print(f"{len(failures)} cases could not run: {next(iter(failures.values()))['error']}")
    if args.json:
        args.json.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    return 0 if passed else 1



if __name__ == "__main__":
    raise SystemExit(main())
