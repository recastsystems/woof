"""Emit woof/data/gfs_pbl/gfs_pbl_constants.cuh from the pinned WRF source.

Every binary64 word the GFS PBL kernel uses is derived here from the text
of WRF v4.6.1 phys/module_gfs_physcons.F (the con_* constants) and
phys/module_bl_gfs.F (MONINP's PARAMETER statements and the default-REAL
literals inside its expressions), by the rule gfortran applies to them:
a literal without a kind is default REAL, so its value is the binary32
rounding of the decimal, widened exactly to binary64; a PARAMETER built
from other PARAMETERs is folded in binary64, one rounding per operation.

The header is therefore the source's constant set, not a retyping of it.
tests/test_gfs_pbl_wrf461_parity.py regenerates it from the copies under
tools/gfs_pbl_wrf461_oracle/upstream/ (SHA-256 pinned below) and compares
bytes, so a constant cannot drift from WRF's word without a test failing.

Usage: python gen_constants.py [upstream_dir] > gfs_pbl_constants.cuh
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PINS = {
    "module_bl_gfs.F":
        "a3898642ee386d60a74af2ef45867e6b5f7df518eef9543ae91586d00372b71c",
    "module_gfs_physcons.F":
        "2b8ef99663ba62748b57c7002f5e5b8c9338f2a825c1d32c113bd4e3bff2eb52",
}


def f32w(text: str) -> np.float64:
    """A default-REAL literal's value: binary32 rounding, widened."""
    return np.float64(np.float32(float(text.lower().replace("d", "e"))))


def physcons(src: str) -> dict[str, np.float64]:
    vals: dict[str, np.float64] = {}
    for name, lit in re.findall(
            r"con_(\w+)\s*=\s*([0-9.]+(?:[eE][+-]?[0-9]+)?)\s*!", src):
        vals[name.lower()] = f32w(lit)
    return vals


def moninp_parameters(src: str) -> dict[str, str]:
    """MONINP's PARAMETER(...) assignments as name -> literal text."""
    body = src[src.index("SUBROUTINE MONINP"):src.index("END SUBROUTINE MONINP")]
    out: dict[str, str] = {}
    for group in re.findall(r"^\s*PARAMETER\s*\((.*)\)\s*$", body, re.M):
        for item in group.split(","):
            name, value = (part.strip() for part in item.split("="))
            out[name.upper()] = value
    return out


def build(upstream: Path) -> str:
    texts = {}
    for name, pin in PINS.items():
        raw = (upstream / name).read_bytes()
        got = hashlib.sha256(raw).hexdigest()
        if got != pin:
            raise SystemExit(f"refusing: {name} is {got}, not WRF v4.6.1's {pin}")
        texts[name] = raw.decode("latin-1")
    con = physcons(texts["module_gfs_physcons.F"])
    par = moninp_parameters(texts["module_bl_gfs.F"])
    d = np.float64
    grav, rd, cp = con["g"], con["rd"], con["cp"]
    rv = con["rv"]
    # con_fvirt = con_rv/con_rd-1. (module_gfs_physcons.F), folded in binary64.
    fv = d(d(rv / rd) - d(1.0))
    lit = {k: f32w(v) for k, v in par.items()
           if re.fullmatch(r"-?[0-9.]+(?:[eE][+-]?[0-9]+)?", v)}
    vk = lit["VK"]
    cfac = lit["CFAC"]
    rows = [
        # name, value, provenance
        ("GFS_G", grav, "PARAMETER(g=grav), grav => con_g = 9.80665e+0"),
        ("GFS_GRAVI", d(d(1.0) / grav), "gravi = 1.0 / grav"),
        ("GFS_FV", fv, "FV => con_fvirt = con_rv/con_rd-1."),
        ("GFS_GOR", d(grav / rd), "PARAMETER(GOR=G/RD)"),
        ("GFS_GOCP", d(grav / cp), "PARAMETER(GOCP=G/CP)"),
        ("GFS_RLAM", lit["RLAM"], "RLAM=" + par["RLAM"]),
        ("GFS_VK", vk, "VK=" + par["VK"]),
        ("GFS_PRMIN", lit["PRMIN"], "PRMIN=" + par["PRMIN"]),
        ("GFS_PRMAX", lit["PRMAX"], "PRMAX=" + par["PRMAX"]),
        ("GFS_DW2MIN", lit["DW2MIN"], "DW2MIN=" + par["DW2MIN"]),
        ("GFS_DKMIN", lit["DKMIN"], "DKMIN=" + par["DKMIN"]),
        ("GFS_DKMAX", lit["DKMAX"], "DKMAX=" + par["DKMAX"]),
        ("GFS_RIMIN", lit["RIMIN"], "RIMIN=" + par["RIMIN"]),
        ("GFS_CFAC", cfac, "CFAC=" + par["CFAC"]),
        ("GFS_PFAC", lit["PFAC"], "PFAC=" + par["PFAC"]
         + " (ZFAC**PFAC lowers to ZFAC*ZFAC)"),
        ("GFS_SFCFRAC", lit["SFCFRAC"], "SFCFRAC=" + par["SFCFRAC"]),
        ("GFS_QMIN", lit["QMIN"], "QMIN=" + par["QMIN"]),
        ("GFS_XKZO", lit["XKZO"], "XKZO=" + par["XKZO"]),
        ("GFS_ZFMIN", lit["ZFMIN"], "ZFMIN=" + par["ZFMIN"]),
        ("GFS_APHI5", lit["APHI5"], "APHI5=" + par["APHI5"]),
        ("GFS_APHI16", lit["APHI16"], "APHI16=" + par["APHI16"]),
        ("GFS_GAMCRT", lit["GAMCRT"], "GAMCRT=" + par["GAMCRT"]),
        ("GFS_GAMCRQ", lit["GAMCRQ"], "GAMCRQ=" + par["GAMCRQ"]),
        ("GFS_CFAC_VK_P1", d(d(cfac * vk) * f32w(".1")),
         "CFAC*VK*.1, folded left to right"),
        # Default-REAL literals inside MONINP/BL_GFS expressions.
        ("GFS_F_P01", f32w(".01"), "(PRSL(i,k)*.01)**ROVCP"),
        ("GFS_F_8", f32w("8."), "1+8.*(-RI)/..."),
        ("GFS_F_1746", f32w("1.746"), "1+1.746*SRI"),
        ("GFS_F_1286", f32w("1.286"), "1+1.286*SRI"),
        ("GFS_F_5", f32w("5."), "(1+5.*RI)**2"),
        ("GFS_F_21", f32w("2.1"), "PRNUM = 1.0 + 2.1*RI"),
    ]
    body = []
    for name, value, why in rows:
        body.append(f"#define {name} ({float(value).hex()})  // {why}")
    table = "\n".join(body) + "\n"
    manifest = hashlib.sha256(table.encode("ascii")).hexdigest()
    head = (
        "// gfs_pbl_constants.cuh -- GENERATED by "
        "tools/gfs_pbl_wrf461_oracle/gen_constants.py; do not edit.\n"
        "// Binary64 words of WRF v4.6.1 module_gfs_physcons.F and MONINP\n"
        "// (module_bl_gfs.F): default-REAL literals are their binary32\n"
        "// rounding widened; PARAMETER expressions fold in binary64.\n"
        f"// source sha256 module_bl_gfs.F {PINS['module_bl_gfs.F']}\n"
        f"// source sha256 module_gfs_physcons.F {PINS['module_gfs_physcons.F']}\n"
        f"// table sha256 {manifest}\n"
        "#ifndef GFS_PBL_CONSTANTS_CUH\n#define GFS_PBL_CONSTANTS_CUH\n")
    return head + table + "#endif\n"


if __name__ == "__main__":
    up = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "upstream"
    sys.stdout.write(build(up))
