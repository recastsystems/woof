#!/usr/bin/env python3
"""Build stock WRF CAM column referees, without altering the physics.

This is a module diagnostic, not a forecast capture. The caller supplies
the input stream. It does not certify the origin of that stream.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COMMON = [
    "phys/module_cam_shr_kind_mod.F", "phys/module_cam_shr_const_mod.F",
    "phys/module_cam_physconst.F", "phys/module_cam_support.F",
    "phys/module_cam_constituents.F", "phys/module_cam_gffgch.F",
    "phys/module_cam_wv_saturation.F", "phys/module_cam_esinti.F",
]
SOURCES = {
    "cu7": COMMON + ["phys/module_cam_cldwat.F", "phys/module_cu_camzm.F",
                     "share/module_model_constants.F", "phys/module_cu_camzm_driver.F"],
    "shcu2": COMMON + ["phys/module_cam_error_function.F",
                       "phys/module_shcu_camuwshcu.F",
                       "phys/module_shcu_camuwshcu_driver.F"],
}
HARNESS = {"cu7": "camzm_wrf461_oracle", "shcu2": "uwshcu_wrf461_oracle"}
FLAGS = [
    "-O2", "-fno-fast-math", "-ffp-contract=off", "-fno-tree-vectorize",
    "-cpp", "-ffree-form", "-ffree-line-length-none", "-fno-range-check",
    "-fallow-argument-mismatch", "-fallow-invalid-boz",
    "-DEM_CORE=1", "-DNMM_CORE=0", "-DCOAMPS_CORE=0", "-DDA_CORE=0",
    "-DEXP_CORE=0", "-DRWORDSIZE=4", "-DIWORDSIZE=4", "-DDWORDSIZE=8",
    "-DLWORDSIZE=4", "-Dwrfmodel", "-DGRIB1", "-DINTIO", "-DLIMIT_ARGS",
    "-DCONFIG_BUF_LEN=65536", "-DMAX_DOMAINS_F=21", "-DMAX_HISTORY=25",
    "-DNMM_NEST=0", "-DNONSTANDARD_SYSTEM_SUBR", "-DUSE_ALLOCATABLES",
]


def verify(source: Path) -> dict:
    pin = json.loads((HERE / "SOURCE.json").read_text())
    for row in pin["files"]:
        path = source / row["path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != row["sha256"]:
            raise ValueError(f"source pin mismatch: {path}: {actual}")
    return pin


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("scheme", choices=SOURCES)
    p.add_argument("out", type=Path)
    p.add_argument("--case", type=Path)
    p.add_argument("--validate-only", action="store_true")
    a = p.parse_args()
    source, out = a.source.resolve(), a.out.resolve()
    pin = verify(source)
    print(f"verified {len(pin['files'])} source pins at {pin['commit']}", flush=True)
    if a.validate_only:
        return
    # Refuse reuse: an old module or object must never enter a new referee.
    out.mkdir(parents=True, exist_ok=False)
    harness = ROOT / "tools" / HARNESS[a.scheme]
    commands: list[list[str]] = []

    def run(command: list[str]) -> None:
        commands.append(command)
        print("$ " + " ".join(command), flush=True)
        subprocess.run(command, cwd=out, check=True)

    compiler = shutil.which("gfortran")
    if compiler is None:
        raise RuntimeError("gfortran is required")
    version = subprocess.check_output([compiler, "--version"], text=True)
    objects = []
    for path in [harness / "stub_wrf.F90"] + [source / f for f in SOURCES[a.scheme]]:
        obj = path.stem + ".o"
        run([compiler, "-c", *FLAGS, "-o", obj, str(path)])
        objects.append(obj)
    if a.scheme == "cu7":
        drivers = [harness / "zmio.F90", harness / "run_camzm.F90"]
        executable = "run_camzm"
    else:
        drivers = [ROOT / "tools/uwpbl_wrf471_oracle/uwio.F90",
                   harness / "run_uwshcu.F90"]
        executable = "run_uwshcu"
    for path in drivers:
        obj = path.stem + ".o"
        run([compiler, "-c", *FLAGS, "-o", obj, str(path)])
        objects.append(obj)
    run([compiler, "-o", executable, *objects])
    receipt = {
        "kind": "stock-native-module-diagnostic",
        "real_forecast_capture": False,
        "scheme": a.scheme, "source_commit": pin["commit"],
        "source_pin": str(HERE / "SOURCE.json"),
        "compiler": version, "commands": commands,
        "physics_modified": False,
        "framework_stubs": str(harness / "stub_wrf.F90"),
    }
    if a.case:
        case = a.case.resolve()
        receipt["input"] = {"path": str(case), "bytes": case.stat().st_size,
                            "sha256": hashlib.sha256(case.read_bytes()).hexdigest()}
        run([str(out / executable), str(case), str(out / "native")])
    (out / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
