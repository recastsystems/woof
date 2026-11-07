"""Grade the CUDA Goddard 4-ice port against the WRF column oracle.

    python tools/gsfcgce4ice_wrf461_oracle/run_gpu.py ORACLE_DIR [--json OUT]
        [--plant]

ORACLE_DIR is a build.sh output directory (cases/case-*.bin and their
*.expected.out words from WRF's own scheme object).  Every case is run
through the production launcher (woof.core.gsfcgce4ice.launch_gsfcgce4ice)
call by call with the oracle's clock, and after every call every graded word
is compared with the word WRF wrote.

Arithmetic arm: GPUWM_WRF_EXACT=1 in the environment selects the strict arm
(the loader compiles with --fmad=false); without it the engine's default
arithmetic runs.  The report says which ran.

--plant recompiles the translation unit with ONE process-term literal
moved by one bit (the Hallett-Mossop splinter count, xnsplnt) and requires
the grade to FAIL: a checker that passes a planted change is not a checker.

Counts: ``differ`` is every word whose bits differ; ``signed_zero`` is the
subset that is +0 against -0. Strict rejects any different word.
Default rejects nonfinite words and residuals above the stated tolerance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import layout  # noqa: E402
from check import compare_words, absolute_tolerance, relative_tolerance, DEFAULT_RTOL

PLANT_FROM = "const float xnsplnt = 370.f,"
PLANT_TO = "const float xnsplnt = 370.00003f,"


def _planted_module():
    import cupy as cp
    from woof.core.kernels import module_options, module_source

    src = module_source("gsfcgce4ice")
    if src.count(PLANT_FROM) != 1:
        raise SystemExit("plant site not found exactly once in the source")
    planted = src.replace(PLANT_FROM, PLANT_TO)
    assert np.float32(370.00003).view(np.uint32) - np.float32(370.0).view(
        np.uint32) == 1
    from woof import wrf_exact
    options = module_options("gsfcgce4ice")
    if wrf_exact.ENABLED:
        options = wrf_exact.effective_options(options)
    mod = cp.RawModule(code=planted, options=options)
    mod.compile()
    return mod


def run_case(case_path: Path, expected_path: Path, module=None):
    import cupy as cp
    from woof.core.fp32_ulp import fp32_ulp_distance
    from woof.core.gsfcgce4ice import launch_gsfcgce4ice

    case = layout.read_input(case_path)
    nz, ncol = case.nz, case.ncol
    expected = layout.read_output(expected_path, ncol, nz, case.nstep)

    def vol(a):
        return cp.asarray(np.ascontiguousarray(a).reshape(a.shape[0], 1, ncol))

    lev = {n: vol(case.level[n]) for n in layout.LEVEL_FIELDS}
    w = vol(case.w)
    sfc = {n: cp.asarray(case.surface[n].reshape(1, ncol))
           for n in layout.SURFACE_FIELDS}
    ht = cp.asarray(case.ht.reshape(1, ncol))
    xland = cp.asarray(case.xland.reshape(1, ncol))
    prec = {n: cp.zeros((nz, 1, ncol), cp.float32) for n in layout.PREC_FIELDS}
    per_field = {}
    for step in range(case.nstep):
        launch_gsfcgce4ice(
            th=lev["th"], qv=lev["qv"], qc=lev["qc"], qr=lev["qr"],
            qi=lev["qi"], qs=lev["qs"], qg=lev["qg"], qh=lev["qh"],
            rho=lev["rho"], pii=lev["pii"], p=lev["p"], z=lev["z"],
            dz8w=lev["dz8w"], w=w, ht=ht, xland=xland,
            refl_10cm=lev["refl_10cm"],
            heating=[lev[n] for n in ("physc", "physe", "physd", "physs",
                                      "physm", "physf")],
            accumulated=[lev[n] for n in ("acphysc", "acphyse", "acphysd",
                                          "acphyss", "acphysm", "acphysf")],
            radii=[lev[n] for n in ("re_cloud_gsfc", "re_rain_gsfc",
                                    "re_ice_gsfc", "re_snow_gsfc",
                                    "re_graupel_gsfc", "re_hail_gsfc")],
            fluxes=[prec[n] for n in layout.PREC_FIELDS],
            rainnc=sfc["rainnc"], rainncv=sfc["rainncv"],
            snownc=sfc["snownc"], snowncv=sfc["snowncv"], sr=sfc["sr"],
            graupelnc=sfc["graupelnc"], graupelncv=sfc["graupelncv"],
            hailnc=sfc["hailnc"], hailncv=sfc["hailncv"],
            dt=float(case.dt), dx=float(case.dx),
            itimestep=int(case.itimestep[step]),
            refl_due=bool(case.diagflag[step]) and case.do_radar_ref == 1,
            module=module)
        got = {}
        for n in layout.LEVEL_FIELDS:
            got[n] = cp.asnumpy(lev[n]).reshape(nz, ncol)
        for n in layout.PREC_FIELDS:
            got[n] = cp.asnumpy(prec[n]).reshape(nz, ncol)
        for n in layout.SURFACE_FIELDS:
            got[n] = cp.asnumpy(sfc[n]).reshape(ncol)
        want = expected[step]
        for n in layout.GRADED_FIELDS:
            a = np.ascontiguousarray(got[n], np.float32)
            b = np.ascontiguousarray(want[n], np.float32)
            grade = compare_words(n, a, b, strict=(os.environ.get('GPUWM_WRF_EXACT') == '1' or module is not None))
            rec = per_field.setdefault(n, {'words': 0, 'differ': 0,
                'signed_zero': 0, 'nonfinite': 0, 'rejected': 0, 'max_ulp': 0,
                'max_abs': 0.0, 'first': None, 'field_hashes': []})
            import hashlib
            rec['field_hashes'].append({'step': step,
                'actual_sha256': hashlib.sha256(a.tobytes()).hexdigest(),
                'expected_sha256': hashlib.sha256(b.tobytes()).hexdigest()})
            for key in ('words', 'differ', 'signed_zero', 'nonfinite', 'rejected'):
                rec[key] += grade[key]
            for key in ('max_ulp', 'max_abs'):
                rec[key] = max(rec[key], grade[key])
            if rec['first'] is None and grade['first'] is not None:
                rec['first'] = {'step': step, **grade['first']}
    return per_field


def main(argv=None) -> int:
    from woof import wrf_exact

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("oracle_dir", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--plant", action="store_true")
    args = ap.parse_args(argv)
    module = _planted_module() if args.plant else None
    cases = sorted((args.oracle_dir / "cases").glob("case-*.bin"))
    if not cases:
        raise SystemExit(f"no cases under {args.oracle_dir}/cases")
    report = {"arm": "strict" if wrf_exact.ENABLED else "default",
              "planted": bool(args.plant), "cases": {}}
    total_words = total_real = total_sz = total_rejected = 0
    worst = 0
    for case_path in cases:
        name = case_path.stem
        per_field = run_case(case_path,
                             case_path.with_name(f"{name}.expected.out"),
                             module)
        report["cases"][name] = per_field
        for n, rec in per_field.items():
            total_words += rec["words"]
            total_sz += rec["signed_zero"]
            total_rejected += rec["rejected"]
            total_real += rec["differ"] - rec["signed_zero"]
            worst = max(worst, rec["max_ulp"])
            if rec["differ"]:
                print(f"{name} {n}: {rec['differ']}/{rec['words']} differ "
                      f"({rec['signed_zero']} signed zero), max_ulp "
                      f"{rec['max_ulp']}, first {rec['first']}")
    report["total"] = {"words": total_words, "nonzero_ulp_words": total_real,
                       "signed_zero_words": total_sz, "max_ulp": worst, "rejected_words": total_rejected}
    print(json.dumps({"arm": report["arm"], "planted": report["planted"],
                      **report["total"]}))
    report['default_tolerance'] = {'relative_by_field': {n: relative_tolerance(n) for n in layout.GRADED_FIELDS}, 'absolute_by_field':
        {n: absolute_tolerance(n) for n in layout.GRADED_FIELDS}}
    if args.json:
        args.json.write_text(json.dumps(report, indent=1) + "\n")
    passed = total_rejected == 0
    if args.plant:
        print("PLANTED CHANGE " + ("NOT CAUGHT" if passed else "CAUGHT"))
        return 1 if passed else 0
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
