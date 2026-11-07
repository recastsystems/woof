"""Every floating constant the Pleim-Xiu column reads, with its WRF spelling.

The CUDA port (woof/core/kernels/pxlsm.cu) never lets NVRTC see an
arithmetic operation on two constants: NVRTC 12.9 folds some such adds and
subtracts with the wrong rounding (measured by the urban port, see
urban_ucm.cu's header), and a constant that reaches the device as a literal
can be propagated into exactly that shape.  So every constant lives in one
``__constant__`` word table, ``pxk_bits``, read through a load, and the
words are what gfortran itself makes of the spellings below -- including
the expressions gfortran folds at compile time (``2.0 * PI * TAUINV``,
``PR0 / KARMAN``, ``EXP(-SALP)`` ...), so the kernel holds the folded word
rather than refolding.

build.sh turns this list into a Fortran program (literals.F90), compiles it
with the as-built flags beside WRF's own module_model_constants, and writes
literals.txt: NAME, the binary32 word, the value.  The kernel's table is
generated from that file (``python constants.py --cuda literals.txt``) and
tests/test_pxlsm_wrf461_parity.py holds the two equal, so neither a typo in
the kernel nor a re-spelling here can pass unnoticed.

Order is the table order.  Names are the kernel's ``K_*`` macro names.
"""

from __future__ import annotations

import sys

#: (name, Fortran expression).  Module parameters of module_sf_pxlsm.F and
#: of its routines are declared in the generated program with WRF's own
#: values (``_PARAMETERS``); module_model_constants is USEd.
CONSTANTS: tuple[tuple[str, str], ...] = (
    # module_model_constants
    ("G", "g"), ("R_D", "r_d"), ("EP_2", "EP_2"), ("SVP1", "SVP1"),
    ("SVP2", "SVP2"), ("SVP3", "SVP3"), ("SVPT0", "SVPT0"),
    ("KARMAN", "KARMAN"), ("STBOLT", "STBOLT"),
    # module_sf_pxlsm parameters and folds
    ("CPD", "CPD"), ("ROVCP", "ROVCP"), ("CRANKP", "CRANKP"),
    ("ONE_MINUS_CRANKP", "1.0 - CRANKP"), ("DENW", "DENW"),
    ("TAUINV", "TAUINV"), ("T2TFAC", "T2TFAC"), ("PI", "PI"),
    ("PR0", "PR0"), ("CZO", "CZO"), ("OZO", "OZO"),
    ("TWO_PI_TAUINV", "2.0 * PI * TAUINV"),
    ("PR0_OVER_KARMAN", "PR0 / KARMAN"),
    ("ALN10", "ALOG(10.0)"),
    ("STBOLT_X1", "STBOLT * (4.0 * CRANKP - 1.0)"),
    ("DTPBLX_PLUS", "DTPBLX + 0.000001"),
    # pxlsm driver literals
    ("C_0", "0.0"), ("C_1", "1.0"), ("C_0P5", "0.5"), ("C_1000", "1000.0"),
    ("C_1EM3", "1.0E-3"), ("C_0P84", "0.84"), ("C_0P001", "0.001"),
    ("C_0P7", "0.7"), ("C_3P670", "3.670"), ("C_290", "290.0"),
    ("C_0P015625", "0.015625"), ("C_100", "100.0"), ("C_8", "8.0"),
    ("C_1EM4", "0.0001"), ("C_22P514", "22.514"), ("C_6P15E3", "6.15E3"),
    ("C_0P622", "0.622"), ("C_4P503", "4.503"), ("C_0P005", "0.005"),
    ("C_1000_INT", "1000"), ("C_0P01", "0.01"),
    # VEGELAND
    ("ZNOTCMN", "ZNOTCMN"), ("ZNOTCMX", "ZNOTCMX"), ("C_9999", "9999.0"),
    ("C_0P10", "0.10"),
    # SURFPX
    ("ZOBS", "ZOBS"), ("GAMAH", "GAMAH"), ("BETAH", "BETAH"),
    ("ONE_MINUS_BETAH", "1.-BETAH"), ("SIGF", "SIGF"),
    ("DWAT_E4", "DWAT*1.E-4"), ("CV", "CV"), ("CT_SNOW", "CT_SNOW"),
    ("CT_IMPERV", "CT_IMPERV"), ("C_2", "2.0"), ("C_5", "5.0"),
    ("C_1P75", "1.75"), ("C_1P718", "1.718"), ("C_3", "3."),
    ("C_14", "14.0"), ("C_10", "10.0"), ("C_2P83E6", "2.83E6"),
    ("C_2P501", "2.501"), ("C_0P00237", "0.00237"), ("C_273P15", "273.15"),
    ("C_1E6", "1.E6"), ("C_M250", "-250.0"), ("C_1EM6", "1.0E-6"),
    ("C_0P2EM3", "0.2E-3"), ("C_1EM11", "1.E-11"), ("C_1P19", "1.19"),
    ("C_5P09", "5.09"), ("C_146", "146."), ("C_1786", "1786."),
    ("C_10P", "10."), ("C_4", "4."), ("C_8P", "8.0"),
    # QFLUX
    ("RSMAX", "RSMAX"), ("F3MIN", "F3MIN"), ("C_130", "130.0"),
    ("C_0P02", "0.02"), ("C_0P07", "0.07"), ("C_0P45", "0.45"),
    ("C_4P566", "4.566"), ("C_M0P0017", "-0.0017"), ("C_M5", "-5.0"),
    ("C_3P0", "3.0"), ("C_302P15", "302.15"), ("C_M0P41", "-0.41"),
    ("C_282P05", "282.05"), ("C_314", "314.0"), ("C_30", "30.0"),
    ("C_1EM5", "0.00001"), ("C_4P0", "4.0"),
    # SMASS
    ("A1MAX", "A1MAX"), ("A2MAX", "A2MAX"), ("B1MAX", "B1MAX"),
    ("B2MAX", "B2MAX"), ("TASSI", "TASSI"), ("RAMIN", "RAMIN"),
    ("WWLTX_PLUS_WFCX", "WWLTX + WFCX"), ("C_1370", "1370.0"),
    ("C_1EM5B", "1.0E-5"),
    # SOILPROP
    ("C_85", "85."), ("C_70", "70."), ("C_M1P08", "-1.08"),
    ("C_494P305", "494.305"), ("C_37P1342EM3", "37.1342E-3"),
    ("C_89P0467EM3", "89.0467E-3"), ("C_0P3496", "0.3496"),
    ("C_0P137", "0.137"), ("C_3P501", "3.501"), ("C_M1P557EM2", "-1.557E-2"),
    ("C_1P441EM2", "1.441E-2"), ("C_4P7021", "4.7021"), ("C_5P58", "5.58"),
    ("C_84P88", "84.88"), ("C_1EM2", "1.0E-2"), ("C_13P815", "13.815"),
    ("C_M0P954", "-0.954"), ("C_5P327", "5.327"), ("C_M1P043", "-1.043"),
    ("C_732P42EM3", "732.42E-3"), ("C_M0P539", "-0.539"),
    ("C_0P134", "0.134"), ("C_3P4", "3.4"), ("C_0P00123", "0.00123"),
    ("C_0P00066", "0.00066"), ("C_0P0405", "0.0405"),
    ("C_0P01B", "0.01"),
    # PXSNOW (RHO_SNOWPACK is a variable set to the integer 450; -O2
    # propagates it, so these three are compile-time words)
    ("LIQSN_RATIO", "DENW / 450.0"), ("CONV_WAT2SNOW", "(DENW / 450.0) / 1000"),
    ("RHO_CS_SNOWPACK", "450.0 * CS_SNOWPACK"), ("C_M2P6", "-SALP"),
    ("EXP_MINUS_SALP", "EXP(-SALP)"),
)

_PARAMETERS = """\
  REAL, PARAMETER :: RD = 287.04, CPD = 1004.67, ROVCP = RD / CPD
  REAL, PARAMETER :: CRANKP = 0.5, DENW = 1000.0, TAUINV = 1.0 / 86400.0
  REAL, PARAMETER :: T2TFAC = 1.0 / 10.0, PI = 3.1415926, PR0 = 0.95
  REAL, PARAMETER :: CZO = 0.032, OZO = 1.E-4, DTPBLX = 40.0
  REAL, PARAMETER :: ZNOTCMN = 5.0, ZNOTCMX = 15.0
  REAL, PARAMETER :: CV = 1.2E-5, ZOBS = 1.5, BH = 15.7, GAMAH = 16.
  REAL, PARAMETER :: BETAH = 5.0, SIGF = 0.5, DWAT = 0.2178
  REAL, PARAMETER :: CT_SNOW = 2.0E-5, CT_IMPERV = 3.268E-6
  REAL, PARAMETER :: RSMAX = 5000.0, FTMIN = 0.0000001, F3MIN = 0.25
  REAL, PARAMETER :: A1MAX = -10.E-5, A2MAX = 1.E-5
  REAL, PARAMETER :: B1MAX = -10.E-3, B2MAX = 1.E-3
  REAL, PARAMETER :: TASSI = 4.6296E-5, RAMIN = 10.0
  REAL, PARAMETER :: WFCX = 0.243, WWLTX = 0.169
  REAL, PARAMETER :: W2SN_CONV = 10.0, CS_SNOWPACK = 2092.0, SALP = 2.6
"""


def fortran_program() -> str:
    lines = ["program literals",
             "  use module_model_constants, only: g, r_d, EP_2, SVP1, SVP2, "
             "SVP3, SVPT0, KARMAN, STBOLT",
             "  implicit none", _PARAMETERS.rstrip("\n"),
             "  open(10, file='literals.txt', status='replace')"]
    for name, expr in CONSTANTS:
        lines.append(f"  write(10, '(A,1X,Z8.8,1X,ES16.9)') '{name}', "
                     f"transfer(real({expr}), 0), real({expr})")
    lines += ["  close(10)", "end program literals", ""]
    return "\n".join(lines)


def cuda_table(literals_path: str) -> str:
    rows = [line.split() for line in open(literals_path) if line.strip()]
    names = [name for name, _ in CONSTANTS]
    if [row[0] for row in rows] != names:
        raise SystemExit("literals.txt does not list CONSTANTS in order")
    out = ["__constant__ unsigned int pxk_bits[] = {"]
    for k, (row, (_, expr)) in enumerate(zip(rows, CONSTANTS)):
        out.append(f"    0x{row[1]}u,  // {k:3d} {row[0]} = {expr}")
    out.append("};")
    for k, row in enumerate(rows):
        out.append(f"#define K_{row[0]} (__uint_as_float(pxk_bits[{k}]))")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    if sys.argv[1:2] == ["--fortran"]:
        sys.stdout.write(fortran_program())
    elif sys.argv[1:2] == ["--cuda"]:
        sys.stdout.write(cuda_table(sys.argv[2]))
    else:
        raise SystemExit("usage: constants.py --fortran | --cuda LITERALS")
