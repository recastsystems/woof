#!/usr/bin/env python3
"""Rebuild the native MPAS-A v8.4.2 diagnostics referee (libdiagnostics.so, libinitial_carriers.so) and the
published real initial columns that tests/test_mpasphys_diagnostics_v842.py and test_mpasphys_numeric_v842.py read.

Every arithmetic statement in the referee is the pinned v8.4.2 source text: this script reads the three native
loops out of the pinned files (refusing if any statement is not found verbatim), writes them into C-callable
wrappers, and compiles them with the pinned framework constants under the strict GNU recipe the native
reference uses (single precision, -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations).

  core_atmosphere/dynamics/mpas_atm_time_integration.F, atm_init_coupled_diagnostics:
      theta_m / rho_zz carriers, exner / exner_base, pressure_p / pressure_base (the cold EOS)
  core_atmosphere/mpas_atm_core.F, atm_compute_output_diagnostics: theta, rho, pressure

The real columns are rho, zz, theta and qv of the published native x1.2562 2023-03-31 12Z init, decoded by a
small Rust reader (decode_columns.rs) as raw little-endian float32 words in the file's (nCells, nVertLevels) order.

Usage: build.py --source MPAS-Model-v8.4.2 [--source-sha256 MANIFEST] --out DIR [--init INIT.nc --columns DIR]
Point MPAS_DIAGNOSTICS_REFERENCE and MPAS_PHYSICS_REFERENCE_DIR at DIR, MPAS_PHYSICS_REAL_COLUMNS at the columns.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLAGS = ["-O3", "-fPIC", "-cpp", "-DSINGLE_PRECISION", "-ffree-form", "-ffree-line-length-none",
         "-fconvert=big-endian", "-ffp-contract=off", "-fno-tree-vectorize", "-fno-unsafe-math-optimizations"]
INTEGRATION = "src/core_atmosphere/dynamics/mpas_atm_time_integration.F"
CORE = "src/core_atmosphere/mpas_atm_core.F"
STATEMENTS = {
    INTEGRATION: [
        "rcv = rgas / (cp-rgas)",
        "p0 = 1.e5",
        "theta_m(k,iCell) = theta(k,iCell) * (1._RKIND + rvord * scalars(index_qv,k,iCell))",
        "rho_zz(k,iCell) = rho(k,iCell) / zz(k,iCell)",
        "exner(k,iCell) = (zz(k,iCell) * (rgas/p0) * (rtheta_p(k,iCell) + rtheta_base(k,iCell)))**rcv",
        "exner_base(k,iCell) = (zz(k,iCell) * (rgas/p0) * (rtheta_base(k,iCell)))**rcv",
        "pressure_p(k,iCell) = zz(k,iCell) * rgas * ( exner(k,iCell) * rtheta_p(k,iCell) + rtheta_base(k,iCell) * (exner(k,iCell) - exner_base(k,iCell)) )",
        "pressure_base(k,iCell) = zz(k,iCell) * rgas * exner_base(k,iCell) * rtheta_base(k,iCell)",
    ],
    CORE: [
        "theta(k,iCell) = theta_m(k,iCell) / (1._RKIND + rvord * scalars(index_qv,k,iCell))",
        "rho(k,iCell) = rho_zz(k,iCell) * zz(k,iCell)",
        "pressure(k,iCell) = pressure_base(k,iCell) + pressure_p(k,iCell)",
    ],
}


def statements(path):
    """Fortran statements of a file: continuations joined, comments and directives dropped, spaces collapsed."""
    out, pending = [], ""
    for line in path.read_text().splitlines():
        code = line.split("!", 1)[0].rstrip()
        if not code.strip():
            continue
        pending += " " + code.strip().lstrip("&").strip()
        if pending.endswith("&"):
            pending = pending[:-1]
            continue
        out.append(re.sub(r"\s+", " ", pending).strip())
        pending = ""
    return out


def norm(text):
    return re.sub(r"\s+", "", text)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-sha256", type=Path, help="pinned sha256sum manifest of the source tree")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--init", type=Path)
    p.add_argument("--columns", type=Path)
    a = p.parse_args()
    src = a.source.resolve()
    receipt = {"schema": "gpuwm-hex.native-diagnostics-reference/v1", "source": str(src), "flags": FLAGS,
               "files": {}, "statements": {}}
    pinned = {}
    if a.source_sha256:
        for line in a.source_sha256.read_text().splitlines():
            digest, _, name = line.partition("  ")
            pinned[name.removeprefix("./")] = digest
    used = [INTEGRATION, CORE, "src/framework/mpas_kind_types.F", "src/framework/mpas_constants.F"]
    for rel in used:
        digest = sha256(src / rel)
        if pinned and pinned.get(rel) != digest:
            sys.exit(f"{rel}: sha256 {digest} is not the pinned {pinned.get(rel)}")
        receipt["files"][rel] = digest
    for rel, wanted in STATEMENTS.items():
        found = {norm(s) for s in statements(src / rel)}
        for text in wanted:
            if norm(text) not in found:
                sys.exit(f"{rel}: statement not found verbatim in the pinned source: {text}")
        receipt["statements"][rel] = wanted
    s = {norm(t): t for rel in STATEMENTS for t in STATEMENTS[rel]}
    i, c = STATEMENTS[INTEGRATION], STATEMENTS[CORE]
    wrappers = f"""! Generated by build.py from the pinned MPAS-A v8.4.2 statements (see diagnostics-reference.json).
module native_diagnostics_reference
   use iso_c_binding, only: c_int, c_float
   use mpas_kind_types, only: RKIND
   use mpas_constants, only: rgas, cp, rvord
   implicit none
   integer, parameter :: index_qv = 1
contains
   subroutine cold(n, zz, rtheta_p, rtheta_base, exner, exner_base, pressure_p, pressure_base) bind(C, name='cold')
      integer(c_int), value :: n
      real(RKIND), intent(in) :: zz(1,n), rtheta_p(1,n), rtheta_base(1,n)
      real(RKIND), intent(out) :: exner(1,n), exner_base(1,n), pressure_p(1,n), pressure_base(1,n)
      real(RKIND) :: rcv, p0
      integer :: k, iCell
      {i[0]}
      {i[1]}
      k = 1
      do iCell=1,n
         {i[4]}
         {i[5]}
      end do
      do iCell=1,n
         {i[6]}
         {i[7]}
      end do
   end subroutine cold

   subroutine output(n, theta_m, scalars, rho_zz, zz, pressure_base, pressure_p, theta, rho, pressure) bind(C, name='output')
      integer(c_int), value :: n
      real(RKIND), intent(in) :: theta_m(1,n), scalars(1,1,n), rho_zz(1,n), zz(1,n), pressure_base(1,n), pressure_p(1,n)
      real(RKIND), intent(out) :: theta(1,n), rho(1,n), pressure(1,n)
      integer :: k, iCell
      k = 1
      do iCell=1,n
         {c[0]}
         {c[1]}
         {c[2]}
      end do
   end subroutine output

   subroutine initialize_carriers(nVertLevels, nCells, rho, zz, theta, scalars, rho_zz, theta_m) bind(C, name='initialize_carriers')
      integer(c_int), value :: nVertLevels, nCells
      real(RKIND), intent(in) :: rho(nVertLevels,nCells), zz(nVertLevels,nCells), theta(nVertLevels,nCells)
      real(RKIND), intent(in) :: scalars(1,nVertLevels,nCells)
      real(RKIND), intent(out) :: rho_zz(nVertLevels,nCells), theta_m(nVertLevels,nCells)
      integer :: k, iCell
      do iCell=1,nCells
         do k=1,nVertLevels
            {i[2]}
            {i[3]}
         end do
      end do
   end subroutine initialize_carriers
end module native_diagnostics_reference
"""
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "native_diagnostics_reference.F90").write_text(wrappers)
    build = out / "build"
    build.mkdir(exist_ok=True)
    for rel in ("src/framework/mpas_kind_types.F", "src/framework/mpas_constants.F"):
        shutil.copy2(src / rel, build / Path(rel).name)
    sources = ["mpas_kind_types.F", "mpas_constants.F", str(out / "native_diagnostics_reference.F90")]
    for name in ("libdiagnostics.so", "libinitial_carriers.so"):
        command = ["gfortran", *FLAGS, "-shared", *sources, "-o", str(out / name)]
        subprocess.run(command, cwd=build, check=True)
        receipt.setdefault("libraries", {})[name] = {"sha256": sha256(out / name), "command": command}
    receipt["gfortran"] = subprocess.run(["gfortran", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    if a.init:
        cols = a.columns.resolve()
        cols.mkdir(parents=True, exist_ok=True)
        exe = out / "decode_columns"
        subprocess.run(["rustc", "--edition=2021", "-C", "opt-level=2", str(HERE / "decode_columns.rs"),
                        "-o", str(exe)], check=True)
        subprocess.run([str(exe), str(a.init), str(cols)], check=True)
        receipt["columns"] = {"init": str(a.init), "init_sha256": sha256(a.init),
                              "files": {f.name: sha256(f) for f in sorted(cols.iterdir())}}
    (out / "diagnostics-reference.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("libraries", "gfortran")}, indent=2))


if __name__ == "__main__":
    main()
