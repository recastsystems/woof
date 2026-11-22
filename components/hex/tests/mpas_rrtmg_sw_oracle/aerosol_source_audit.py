"""Compare MPAS aerosol sources with the archived WOOF-ArWen HRRR authority.

Full routine differences remain in the receipt. The two scoped equivalences
remove only MPAS module imports / IMPLICIT NONE, or explicit RKIND spelling;
they do not remove numerical statements or constants.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re


def routine(path, name):
    lines = Path(path).read_text().splitlines(keepends=True)
    start = None
    for index, line in enumerate(lines):
        code = line.split("!", 1)[0]
        if start is None and re.search(r"\b(?:subroutine|function)\s+" + name + r"\s*\(", code, re.I):
            start = index
        elif start is not None and re.search(r"\bend\s+(?:subroutine|function)\s+" + name + r"\b", code, re.I):
            return "".join(lines[start:index+1]), start+1, index+1
    raise ValueError(f"No complete {name} in {path}")


def normalized(text, *, drop_framework=False, rkind_as_real=False):
    lines = [line.split("!", 1)[0].lower() for line in text.splitlines()]
    if drop_framework:
        lines = [line for line in lines if not re.match(r"\s*(?:use\b|implicit\s+none\b)", line)]
    code = re.sub(r"\s+", "", "".join(lines))
    if rkind_as_real:
        code = code.replace("real(kind=rkind)", "real")
    return code


def audit(mpas_root, hrrr_driver, hrrr_extract, receipt):
    root = Path(mpas_root)/"src/core_atmosphere/physics"
    aerosol = root/"physics_wrf/module_ra_rrtmg_sw_aerosols.F"
    pairs = [("gt_aod", root/"physics_wrf/module_mp_thompson_aerosols.F", hrrr_driver),
             ("rslf", root/"mpas_atmphys_functions.F", hrrr_extract)]
    pairs += [(name, aerosol, hrrr_extract) for name in (
        "calc_aerosol_rrtmg_sw", "calc_spectral_aod_rrtmg_sw", "calc_spectral_ssa_rrtmg_sw",
        "calc_spectral_asy_rrtmg_sw", "aod_profiler", "calc_relative_humidity")]
    rows, diffs = [], []
    for name, mpas, wrf in pairs:
        m, mf, ml = routine(mpas, name)
        w, wf, wl = routine(wrf, name)
        row = dict(routine=name, mpas=str(mpas), hrrr=str(wrf),
                   mpas_lines=[mf, ml], hrrr_lines=[wf, wl],
                   mpas_routine_sha256=hashlib.sha256(m.encode()).hexdigest(),
                   hrrr_routine_sha256=hashlib.sha256(w.encode()).hexdigest(),
                   text_identical=m==w, code_identical=normalized(m)==normalized(w))
        if name == "gt_aod":
            row["identity_scope"] = "All statements, declarations and table constants after USE and IMPLICIT NONE removal"
            row["scoped_code_identical"] = normalized(m, drop_framework=True)==normalized(w, drop_framework=True)
        if name == "rslf":
            row["identity_scope"] = "All statements and constants with REAL(KIND=RKIND) treated as REAL; SINGLE_PRECISION RKIND=4"
            row["scoped_code_identical"] = normalized(m, rkind_as_real=True)==normalized(w, rkind_as_real=True)
        rows.append(row)
        diffs.extend(difflib.unified_diff(w.splitlines(keepends=True), m.splitlines(keepends=True),
                                       fromfile="HRRR:"+name, tofile="MPAS:"+name))
    receipt = Path(receipt)
    result = dict(origin="NOAA HRRR v4.1.21", origin_commit="40ee6058c2fc6624cbfbbe8cf1c20c59e6a45827",
                  hrrr_extract_sha256=hashlib.sha256(Path(hrrr_extract).read_bytes()).hexdigest(),
                  hrrr_driver_sha256=hashlib.sha256(Path(hrrr_driver).read_bytes()).hexdigest(), routines=rows)
    receipt.write_text(json.dumps(result, indent=2)+"\n")
    receipt.with_suffix(".diff").write_text("".join(diffs))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("mpas_root", "hrrr_driver", "hrrr_extract", "receipt"):
        parser.add_argument(name)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))
