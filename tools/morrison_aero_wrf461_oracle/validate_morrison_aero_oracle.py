"""Run the mp_physics=40 GPU port on the WRF column oracle and measure it.

Every expected word in the fixture was written by WRF v4.6.1's own
MP_MORR_TWO_MOMENT_AERO (tools/morrison_aero_wrf461_oracle/build.sh).  This
module feeds the same inputs to woof.core.morrison_aero.launch_morrison_aero
and reports, per output field, the number of differing words and the
maximum total-order FP32 ULP distance (woof.core.fp32_ulp).

usage: python validate_morrison_aero_oracle.py [VARIANT-TAG ...]
"""

from __future__ import annotations

import csv
import gzip
import io
import sys
from pathlib import Path

import numpy as np

DEFAULT_FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles"
                       / "morrison" / "aero_oracle")

#: (variant, tag) pairs committed in DEFAULT_FIXTURE_DIR.  ``pinned`` is the
#: byte-unmodified module; ``t0fix`` carries declared divergence D1 and is
#: committed only where it differs from pinned (aercu_opt = 2).
FIXTURES = (
    ("pinned", "r1a0dt20"), ("pinned", "r0a0dt20"), ("pinned", "r1a0dt60"),
    ("pinned", "r1a1dt20"), ("pinned", "r1a2dt20"), ("pinned", "r0a2dt30"),
    ("t0fix", "r1a2dt20"), ("t0fix", "r0a2dt30"),
)

LEVEL_INPUTS = ("th", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "ns", "nr",
                "ng", "nc", "pii", "p", "dz", "w", "kzh_up", "qrcuten",
                "qscuten", "qicuten") + tuple(f"aero{s}" for s in range(1, 11))
STATE_OUTPUTS = ("th", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "ns", "nr",
                 "ng", "nc")
SURFACE = ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
           "graupelncv", "sr")


def _rows(path: Path) -> list[dict[str, str]]:
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    return list(csv.DictReader(io.StringIO(raw.decode("ascii"))))


def _word(text: str) -> np.uint32:
    return np.uint32(int(text, 16))


def load_fixture(variant: str, tag: str, fixture_dir: Path = DEFAULT_FIXTURE_DIR):
    """Inputs and WRF outputs as float32 arrays, (nz, 1, ncol) and (1, ncol)."""
    levels = _rows(fixture_dir / f"{variant}-{tag}-levels.csv.gz")
    surface = _rows(fixture_dir / f"{variant}-{tag}-surface.csv.gz")
    ncol = max(int(r["col"]) for r in levels)
    nz = max(int(r["k"]) for r in levels)

    def field(rows, name, three_d=True):
        out = np.zeros((nz, 1, ncol) if three_d else (1, ncol), dtype=np.uint32)
        for r in rows:
            c = int(r["col"]) - 1
            if three_d:
                out[int(r["k"]) - 1, 0, c] = _word(r[name])
            else:
                out[0, c] = _word(r[name])
        return out.view(np.float32)

    first = levels[0]
    meta = {"rimed": int(first["rimed"]), "aercu_opt": int(first["aercu_opt"]),
            "dt": float(np.uint32(int(first["dt"], 16)).view(np.float32)),
            "nz": nz, "ncol": ncol, "activation_temperature": "wrf" if variant == "pinned" else "melting-point"}
    inputs = {name: field(levels, f"{name}_in") for name in LEVEL_INPUTS}
    inputs.update({name: field(surface, f"{name}_in", False) for name in SURFACE})
    expected = {name: field(levels, f"{name}_out") for name in STATE_OUTPUTS}
    expected["refl_10cm"] = field(levels, "refl_10cm_out")
    expected["mskf_refl_10cm"] = field(levels, "mskf_refl_10cm_out")
    for name in ("efcg", "efig", "efsg", "wact"):
        expected[name] = field(levels, f"{name}_out")
    for s in range(1, 8):
        expected[f"ccn{s}"] = field(levels, f"ccn{s}_out")
    expected.update({name: field(surface, f"{name}_out", False) for name in SURFACE})
    return meta, inputs, expected


def run_port(meta, inputs):
    """Run the GPU port on one fixture; returns host float32 outputs."""
    import cupy as cp
    from woof.core.morrison_aero import launch_morrison_aero
    from woof.core.morrison_aero import launch_refl_morrison_aero

    d = {name: cp.asarray(np.ascontiguousarray(v)) for name, v in inputs.items()}
    shape = d["th"].shape
    aero = cp.ascontiguousarray(cp.stack([d[f"aero{s}"] for s in range(1, 11)]))
    ccn = cp.zeros((7,) + shape, dtype=cp.float32)
    rad = {name: cp.zeros(shape, dtype=cp.float32)
           for name in ("efcg", "efig", "efsg", "wact")}
    aercu = meta["aercu_opt"]
    scratch = launch_morrison_aero(
        d["th"], d["qv"], d["qc"], d["qr"], d["qi"], d["qs"], d["qg"],
        d["ni"], d["ns"], d["nr"], d["ng"], d["nc"], d["pii"], d["p"],
        d["dz"], d["rainnc"], d["rainncv"], d["snownc"], d["snowncv"],
        d["graupelnc"], d["graupelncv"], d["sr"], meta["dt"],
        morr_rimed_ice=meta["rimed"], aercu_opt=aercu, activation_temperature=meta["activation_temperature"], w=d["w"],
        kzh_up=d["kzh_up"], aerocu=aero, qrcuten=d["qrcuten"],
        qscuten=d["qscuten"], qicuten=d["qicuten"],
        ccn=ccn if aercu > 0 else None,
        **({k: v for k, v in rad.items()} if aercu > 0 else {}))
    refl = cp.zeros(shape, dtype=cp.float32)
    temperature = cp.ascontiguousarray(scratch[0])
    launch_refl_morrison_aero(d["qv"], d["qr"], d["nr"], d["qs"], d["ns"],
                             d["qg"], d["ng"], temperature, d["p"], refl,
                             morr_rimed_ice=meta["rimed"])
    out = {name: cp.asnumpy(d[name]) for name in STATE_OUTPUTS + SURFACE}
    out["refl_10cm"] = cp.asnumpy(refl)
    # The oracle supplies zero convective mass/number and zero CU_UAF.
    out["mskf_refl_10cm"] = cp.asnumpy(refl)
    out.update({name: cp.asnumpy(v) for name, v in rad.items()})
    for s in range(7):
        out[f"ccn{s + 1}"] = cp.asnumpy(ccn[s])
    return out


def compare(expected, got) -> dict[str, dict[str, int]]:
    """Per field: differing words and maximum total-order ULP distance."""
    from woof.core.fp32_ulp import fp32_ulp_distance
    report = {}
    for name, value in got.items():
        want = expected[name]
        same = want.view(np.uint32) == value.view(np.uint32)
        ulp = fp32_ulp_distance(value, want)
        report[name] = {"differ": int((~same).sum()),
                        "max_ulp": int(np.max(ulp)) if ulp.size else 0,
                        "n": int(same.size),
                        "max_abs": float(np.max(np.abs(value.astype(np.float64)-want.astype(np.float64))))}
    return report


def measure(variant: str, tag: str, fixture_dir: Path = DEFAULT_FIXTURE_DIR):
    meta, inputs, expected = load_fixture(variant, tag, fixture_dir)
    return meta, compare(expected, run_port(meta, inputs))


def main(argv: list[str]) -> int:
    pairs = [tuple(a.split("-", 1)) for a in argv] or list(FIXTURES)
    worst = 0
    for variant, tag in pairs:
        meta, report = measure(variant, tag)
        bad = {k: v for k, v in report.items() if v["differ"]}
        line = ", ".join(f"{k} {v['differ']}/{v['n']} max {v['max_ulp']}"
                         for k, v in bad.items()) or "all words identical"
        print(f"{variant}-{tag} (rimed {meta['rimed']}, aercu_opt "
              f"{meta['aercu_opt']}, dt {meta['dt']:g}): {line}")
        worst = max([worst] + [v["max_ulp"] for v in report.values()])
    print(f"worst max_ulp {worst}")
    return int(worst != 0)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
