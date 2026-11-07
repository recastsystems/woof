"""Run the built oracle on every configuration and package the fixtures.

usage: package.py BUILD_DIR FIXTURE_DIR

For each configuration in make_cases.CONFIGS: generate the input, run the
scalar build (the referee: WRF's own -O2 -fno-fast-math -ffp-contract=off
-fno-tree-vectorize flags) and the -O0 control, record every word where the
two builds differ, and copy the input and the referee's words into
FIXTURE_DIR with a SHA-256 manifest.  The -O0 control is evidence, never
the referee: WRF ships the -O2 build.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_cases  # noqa: E402

OUT_NAMES = (*make_cases.STATE, "u10", "v10", "chs_sea", "chs2_sea",
             "cpm_sea", "cqs2_sea", "flhc_sea", "flqc_sea", "hfx_sea",
             "lh_sea", "qfx_sea", "qgh_sea", "qsfc_sea", "tsk_sea")


def _words(path: Path) -> np.ndarray:
    head = np.fromfile(path, dtype="<i4", count=4)
    return np.fromfile(path, dtype="<u4")[4:].reshape(head[2], head[3],
                                                     head[1])


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: package.py BUILD_DIR FIXTURE_DIR", file=sys.stderr)
        return 2
    build, fixtures = Path(argv[1]), Path(argv[2])
    cases = build / "cases"
    cases.mkdir(parents=True, exist_ok=True)
    fixtures.mkdir(parents=True, exist_ok=True)
    index, control = {}, {}
    for name, spec in make_cases.CONFIGS.items():
        index[name] = make_cases.write_config(cases, name, spec)
        source = cases / f"{name}.in.bin"
        for variant in ("scalar", "O0"):
            subprocess.run([str(build / variant / "run_pxsfclay"),
                            str(source),
                            str(cases / f"{name}.{variant}.out.bin")],
                           check=True)
        a = _words(cases / f"{name}.scalar.out.bin")
        b = _words(cases / f"{name}.O0.out.bin")
        rows = []
        for step, field, col in np.argwhere(a != b):
            rows.append({"step": int(step), "field": OUT_NAMES[field],
                         "column": int(col),
                         "family": index[name]["families"][col],
                         "scalar": f"{int(a[step, field, col]):08x}",
                         "O0": f"{int(b[step, field, col]):08x}"})
        control[name] = rows
        shutil.copyfile(source, fixtures / f"{name}.in.bin")
        shutil.copyfile(cases / f"{name}.scalar.out.bin",
                        fixtures / f"{name}.out.bin")
    (fixtures / "cases-index.json").write_text(
        json.dumps(index, indent=1) + "\n")
    (fixtures / "o0-control.json").write_text(
        json.dumps(control, indent=1) + "\n")
    shutil.copyfile(build / "toolchain.txt", fixtures / "toolchain.txt")
    shutil.copyfile(build / "extract-receipt.json",
                    fixtures / "extract-receipt.json")
    names = sorted(p.name for p in fixtures.iterdir()
                   if p.name != "MANIFEST.json")
    manifest = {
        "referee": "WRF v4.6.1 d66e442 phys/module_sf_pxsfclay.F and "
                   "module_surface_driver.F pxsfclay_seaice_wrapper/"
                   "get_local_ice_tsk, gfortran -O2 -fno-fast-math "
                   "-ffp-contract=off -fno-tree-vectorize",
        "configurations": list(make_cases.CONFIGS),
        "sha256": {n: hashlib.sha256((fixtures / n).read_bytes()).hexdigest()
                   for n in names},
    }
    (fixtures / "MANIFEST.json").write_text(json.dumps(manifest, indent=1)
                                            + "\n")
    print(json.dumps({n: len(r) for n, r in control.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
