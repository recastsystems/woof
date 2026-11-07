#!/usr/bin/env python3
"""Build the unmodified, pinned public WRF NTU module, with a receipt.

The diagnostic initialization driver is analytic. It is an undefined-read
witness, not the real-input GPU replay required for selector admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "source.json").read_text())
FLAGS = (
    "-O2", "-fno-fast-math", "-ffp-contract=off", "-fno-tree-vectorize",
    "-ffree-form", "-ffree-line-length-none", "-g", "-fcheck=bounds",
    "-finit-real=snan", "-ffpe-trap=invalid,zero,overflow",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(source: Path, licence: Path) -> None:
    # Check the licence before processing or compiling the source.
    for path, spec in ((licence, MANIFEST["licence"]),
                       (source, MANIFEST["module"])):
        actual = digest(path)
        if actual != spec["sha256"]:
            raise ValueError(f"{path}: expected {spec['sha256']}, got {actual}")


def build(source: Path, licence: Path, out: Path, compiler: str) -> dict:
    source, licence, out = source.resolve(), licence.resolve(), out.resolve()
    verify(source, licence)
    out.mkdir(parents=True, exist_ok=True)
    stub = HERE.parent / "ntu_wrf461_oracle" / "stub_wrf_error.F90"
    commands = [
        [compiler, *FLAGS, "-c", str(stub), "-o", "stub_wrf_error.o"],
        [compiler, *FLAGS, "-cpp", "-DRWORDSIZE=4", "-DDWORDSIZE=8", "-c",
         str(source), "-o", "module_mp_ntu.o"],
        [compiler, *FLAGS, "-c", str(HERE / "init_witness.F90"),
         "-o", "init_witness.o"],
        [compiler, *FLAGS, "stub_wrf_error.o", "module_mp_ntu.o",
         "init_witness.o", "-o", "init_witness"],
    ]
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for command in commands:
        subprocess.run(command, cwd=out, env=environment, check=True)
    receipt = {
        "source": MANIFEST,
        "source_path": str(source), "licence_path": str(licence),
        "compiler": subprocess.check_output([compiler, "--version"], text=True),
        "commands": commands, "cwd": str(out),
        "driver_sha256": digest(HERE / "init_witness.F90"),
        "executable_sha256": digest(out / "init_witness"),
        "input_origin": "analytic initialization only, no forecast input",
        "real_input_replay_words": 0,
    }
    (out / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("licence", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--compiler", default="gfortran")
    args = ap.parse_args()
    build(args.source, args.licence, args.out, args.compiler)


if __name__ == "__main__":
    main()
