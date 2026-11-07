#!/usr/bin/env python3
"""Build isolated, verbatim WRF v4 ENERGY blocks. No forecast proof claim."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

LSM_SHA256 = "bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282"
NOAHMP_PIN = "848f54ad3d28c4303151fe5ad83724e232694422"
FLAGS = ["-shared", "-fPIC", "-O0", "-ffp-contract=off", "-fno-fast-math",
         "-fno-tree-vectorize", "-ffree-form", "-ffree-line-length-none"]


def build(lsm: Path, work: Path) -> Path:
    original = lsm.read_bytes()
    if hashlib.sha256(original).hexdigest() != LSM_SHA256:
        raise ValueError("refusing unpinned module_sf_noahmplsm.F")
    text = original.decode("utf-8")
    first = text.index("     BTRAN = 0.0\n", text.index("  SUBROUTINE ENERGY"))
    last = text.index("! soil surface resistance for ground evap.", first)
    block = text[first:last]
    rsf_first = text.index("     BEVAP =", last)
    rsf_last = text.index("! set psychrometric constant", rsf_first)
    rsf_block = text[rsf_first:rsf_last]
    assert "IF(OPT_BTR == 2)" in block and "IF(OPT_BTR == 3)" in block
    constants = "\n".join(line for line in text.splitlines()
                          if ":: MPE    =" in line or ":: PSIWLT =" in line)
    assert len(constants.splitlines()) == 2
    wrapper = '''module noahmp_option_referee
use iso_c_binding
implicit none
type soil_parameters
 integer :: nroot
 real :: smcwlt(4), smcref(4), psisat(4), smcmax(4), bexp(4)
 real :: rsurf_exp, rsurf_snow
 logical :: urban_flag
end type
contains
subroutine noahmp_btran(opt_btr,nroot,sh2o,smcwlt,smcref,psisat,smcmax,bexp, &
                        zsoil,dzsnso,btran,btrani) bind(C)
 integer(c_int), value :: opt_btr, nroot
 real(c_float), intent(in) :: sh2o(4), smcwlt(4), smcref(4), psisat(4), &
                              smcmax(4), bexp(4), zsoil(4), dzsnso(-2:4)
 real(c_float), intent(out) :: btran
 real(c_float), intent(inout) :: btrani(4)
 type(soil_parameters) :: parameters
 integer :: ist, iz
 real :: gx, psi
CONSTANTS
 parameters%nroot=nroot
 parameters%smcwlt=smcwlt
 parameters%smcref=smcref
 parameters%psisat=psisat
 parameters%smcmax=smcmax
 parameters%bexp=bexp
 ist=1
BLOCK
end subroutine
subroutine noahmp_rsurf(opt_rsf,urban,inputs,outputs) bind(C)
 integer(c_int), value :: opt_rsf, urban
 real(c_float), intent(in) :: inputs(11)
 real(c_float), intent(out) :: outputs(2)
 type(soil_parameters) :: parameters
 integer :: ist
 real :: sh2o(4), zsoil(4), fsno, snowh, tg, bevap, l_rsurf, d_rsurf, rsurf, rhsur, psi
RSF_CONSTANTS
 sh2o=0.0
 zsoil=0.0
 sh2o(1)=inputs(1)
 parameters%smcmax(1)=inputs(2)
 parameters%smcwlt(1)=inputs(3)
 parameters%bexp(1)=inputs(4)
 parameters%psisat(1)=inputs(5)
 zsoil(1)=inputs(6)
 parameters%rsurf_exp=inputs(7)
 parameters%rsurf_snow=inputs(8)
 parameters%urban_flag=urban/=0
 fsno=inputs(9)
 snowh=inputs(10)
 tg=inputs(11)
 ist=1
RSF_BLOCK
 outputs(1)=rsurf
 outputs(2)=rhsur
end subroutine
end module
'''.replace("RSF_CONSTANTS", "\n".join(
        line for line in text.splitlines() if ":: GRAV " in line or ":: RW " in line))
    wrapper = wrapper.replace("RSF_BLOCK", rsf_block).replace("CONSTANTS", constants).replace("BLOCK", block)
    work.mkdir(parents=True, exist_ok=True)
    source = work / "btran_native.F90"
    source.write_text(wrapper)
    library = work / "libnoahmp_options.so"
    command = ["gfortran", *FLAGS, "-J", str(work.resolve()),
               str(source.resolve()), "-o", str(library.resolve())]
    subprocess.run(command, check=True, cwd=work)
    symbols = subprocess.check_output(["nm", "-u", str(library)], text=True)
    if "_ZGV" in symbols:
        raise RuntimeError("native referee unexpectedly calls vector libm")
    (work / "undefined-symbols.txt").write_text(symbols)
    provenance = {
        "scope": "isolated WRF ENERGY BTRAN/RSURF blocks; constructed inputs; not a real WRF run",
        "wrf_pin": "d66e442fccc04111067e29274c9f9eaccc3cef28",
        "noahmp_pin": NOAHMP_PIN, "source_sha256": LSM_SHA256,
        "source_lines": [text[:first].count("\n") + 1, text[:last].count("\n")],
        "block_sha256": hashlib.sha256(block.encode("ascii")).hexdigest(),
        "rsurf_source_lines": [text[:rsf_first].count("\n") + 1, text[:rsf_last].count("\n")],
        "rsurf_block_sha256": hashlib.sha256(rsf_block.encode("ascii")).hexdigest(),
        "licence": "WRF v4 public-domain distribution; see LICENSE.txt at the WRF pin",
        "command": command,
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "native_undefined": "BTRANI above NROOT is not assigned; compare only active roots",
    }
    (work / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return library


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lsm", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    args = parser.parse_args()
    print(build(args.lsm.resolve(), args.work.resolve()))
