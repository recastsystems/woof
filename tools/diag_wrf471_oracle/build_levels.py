#!/usr/bin/env python3
"""Build unchanged WRF diagnostic leaves with strict Fortran arithmetic."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((HERE / "sources.json").read_text())
    hashes = {}
    for name in ("module_model_constants.F", "module_diag_pld.F", "module_diag_zld.F"):
        path = HERE / "upstream" / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        row = next(r for r in manifest["sources"] if Path(r["path"]).name == name)
        if digest != row["sha256"]:
            raise ValueError(f"source hash mismatch: {path}")
        hashes[name] = digest
    command = ["gfortran", "-cpp", "-DNMM_CORE=0", "-O0", "-fPIC", "-shared",
               "-ffp-contract=off", "-fno-fast-math", "-ffree-form", "-ffree-line-length-none",
               *(str(HERE / "upstream" / n) for n in hashes),
               str(HERE / "levels_ref.f90"), "-o", str(output / "levels_ref.so")]
    subprocess.run(command, cwd=output, check=True)
    receipt = {"command": command, "source_sha256": hashes,
               "wrf": manifest["wrf"], "compiler": subprocess.check_output(["gfortran", "--version"], text=True),
               "libc": platform.libc_ver(), "adapter_sha256": hashlib.sha256((HERE / "levels_ref.f90").read_bytes()).hexdigest(),
               "library_sha256": hashlib.sha256((output / "levels_ref.so").read_bytes()).hexdigest()}
    (output / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(output / "build.json")


if __name__ == "__main__":
    main()
