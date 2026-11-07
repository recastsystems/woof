"""The TEMF pair's constants: one table, one hash, one generator.

WRF v4.6.1's TEMF PBL (phys/module_bl_temf.F, bl_pbl_physics = 10) and
TEMF surface layer (phys/module_sf_temfsfclay.F, sf_sfclay_physics = 10)
take three kinds of constant: the model constants their drivers pass
(share/module_model_constants.F), the PARAMETERs of the two modules
(module_bl_temf.F:354-374, module_sf_temfsfclay.F:198-212) and the decimal
literals in their statements.  Every one whose decimal is not an exactly
representable float32 is a row here: its Fortran text, and the float32
word gfortran makes of it.  A PARAMETER expression is folded at compile
time with each operation rounded to float32, which is what
:func:`fold` reproduces.

The CUDA kernel (woof/core/kernels/temf.cu) reads these words from its
``TEMFC`` table, which :func:`cuda_table` writes; ``TEMF_CONSTANTS_SHA256``
pins the table, and tests/test_temf_constants.py re-derives every word with
:func:`fold`, re-generates the CUDA block and checks both against the
kernel source.  tools/temf_wrf461_oracle/check_constants.py compiles the
same expressions with gfortran and checks gfortran's words against these.
"""
from __future__ import annotations

import ast
import hashlib
import sys

import numpy as np

F32 = np.float32

#: (name, Fortran expression, float32 word).  Names may refer to earlier
#: rows.  The order is the CUDA table's index order; append only.
TEMF_CONSTANTS: tuple[tuple[str, str, int], ...] = (
    # share/module_model_constants.F
    ("G", "9.81", 0x411CF5C3),
    ("R_D", "287.", 0x438F8000),
    ("CP", "7.*R_D/2.", 0x447B2000),
    ("R_V", "461.6", 0x43E6CCCD),
    ("CPV", "4.*R_V", 0x44E6CCCD),
    ("RCP", "R_D/CP", 0x3E924925),
    ("XLV", "2.5E6", 0x4A189680),
    ("EP2", "R_D/R_V", 0x3F1F2B09),
    ("KARMAN", "0.4", 0x3ECCCCCD),
    ("SVP1", "0.6112", 0x3F1C779A),
    ("SVP2", "17.67", 0x418D5C29),
    ("SVP3", "29.65", 0x41ED3333),
    ("SVPT0", "273.15", 0x43889333),
    # module_bl_temf.F:354-374 (the surface layer repeats ftau0, fth0,
    # Cf, CN, Ceps and PrT0 with the same values, :202-212)
    ("VISC_TEMF", "1.57e-4", 0x3924A05E),
    ("CONDUC_TEMF", "1.57e-4 / 0.733", 0x396097BA),
    ("TEMIN", "1e-3", 0x3A83126F),
    ("FTAU0", "0.17", 0x3E2E147B),
    ("FTH0", "0.145", 0x3E147AE1),
    ("CF", "0.185", 0x3E3D70A4),
    ("CEPS", "0.070", 0x3D8F5C29),
    ("PRT0", "CEPS/CEPS * FTAU0**2 / 2. / FTH0**2", 0x3F2FF165),
    ("CM", "0.03", 0x3CF5C28F),
    ("CDELT", "0.006", 0x3BC49BA6),
    # Folded constant subexpressions of module_bl_temf.F statements.
    ("RATIO_POS",
     "CEPS**2.*FTAU0**2./(2.*CEPS**2.*FTH0**2.)", 0x3F2FF165),   # :595
    ("RATIO_NEG",
     "CEPS**2.*FTAU0**2./(-2.*CEPS**2.*FTH0**2.)", 0xBF2FF165),  # :600
    ("THREE_QUARTERS", "3./4.", 0x3F400000),                     # :596
    ("ONE_QUARTER", "1./4.", 0x3E800000),                        # :596
    ("INV_LASYMP", "1./200.0", 0x3BA3D70A),                      # :610
    ("KHCONV", "FTAU0**2. / CEPS / PRT0", 0x3F19C868),           # :647
    ("ONE_THIRD", "1./3.", 0x3EAAAAAB),                          # :572
    ("TWO_THIRDS", "2./3.", 0x3F2AAAAB),                         # :1087
    ("PI_LIT", "3.1415926", 0x40490FDA),                         # :846
    ("HALF_PI_LIT", "3.1415926/2", 0x3FC90FDA),                  # :846
    # Decimal literals in module_bl_temf.F statements.
    ("K0P608", "0.608", 0x3F1BA5E3),
    ("TINY15", "1e-15", 0x26901D7D),
    ("K0P002", "0.002", 0x3B03126F),
    ("K0P33", "0.33", 0x3EA8F5C3),
    ("K0P622", "0.622", 0x3F1F3B64),
    ("K0P9", "0.9", 0x3F666666),
    ("K0P05", "0.05", 0x3D4CCCCD),
    ("K0P36", "0.36", 0x3EB851EC),
    ("K1P55", "1.55", 0x3FC66666),
    ("K0P61", "0.61", 0x3F1C28F6),
    ("K0P286", "0.286", 0x3E926E98),
    ("K2P45EM6", "2.45e-6", 0x36246AAE),
    ("K1004P67", "1004.67", 0x447B2AE1),
    ("K1P61", "1.61", 0x3FCE147B),
    ("K0P7", "0.7", 0x3F333333),
    ("K0P3", "0.3", 0x3E99999A),
    ("K0P001", "0.001", 0x3A83126F),
    # rsat, module_bl_temf.F:1293-1303
    ("RSAT_C0", "0.6105851e+3", 0x4418A572),
    ("RSAT_C1", "0.4440316e+2", 0x42319CD6),
    ("RSAT_C2", "0.1430341e+1", 0x3FB7156A),
    ("RSAT_C3", "0.2641412e-1", 0x3CD8626D),
    ("RSAT_C4", "0.2995057e-3", 0x399D06F9),
    ("RSAT_C5", "0.2031998e-5", 0x36085D76),
    ("RSAT_C6", "0.6936113e-8", 0x31EE52B2),
    ("RSAT_C7", "0.2564861e-11", 0x2C347C6E),
    ("RSAT_C8", "-0.3704404e-13", 0xA926D4DF),
    # module_sf_temfsfclay.F
    ("CONDUC_SFC", "1.57e-5 / 0.733", 0x37B3AC94),               # :199
    ("K0P2", "0.2", 0x3E4CCCCD),                                 # :280, :285
    ("K1EM2", "1e-2", 0x3C23D70A),                               # :340
    ("K0P8", "0.8", 0x3F4CCCCD),                                 # :380
)


def _literal(text: str) -> np.float32:
    """A Fortran default-REAL literal as gfortran reads it (one rounding)."""
    from fractions import Fraction
    value = Fraction(text.lower().replace("d", "e"))
    lo = F32(float(value))
    # float(Fraction) is correctly rounded to binary64; the step to float32
    # can double-round, so settle it exactly against the two neighbours.
    best = None
    for cand in (np.nextafter(lo, F32(-np.inf)), lo,
                 np.nextafter(lo, F32(np.inf))):
        err = abs(Fraction(float(cand)) - value)
        if best is None or err < best[0] or (
                err == best[0] and (int(cand.view(np.uint32)) & 1) == 0):
            best = (err, cand)
    return F32(best[1])


def fold(expr: str, names: dict | None = None) -> np.float32:
    """Fold a Fortran REAL constant expression the way gfortran does.

    Operations are evaluated left to right as the parse tree says, each
    rounded to float32.  ``x**2`` and ``x**2.`` are ``x*x`` (an exact
    square, correctly rounded, is what MPFR's folding gives); any other
    power is refused, because no row here needs one.
    """
    names = dict(names or {})
    tree = ast.parse(expr.replace("**", "^POW^").replace("^POW^", "**"),
                     mode="eval")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            return _literal(repr(node.value) if not isinstance(node.value, str)
                            else node.value)
        if isinstance(node, ast.Name):
            return F32(names[node.id])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return F32(-ev(node.operand))
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow):
                base = ev(node.left)
                exp = node.right
                if (isinstance(exp, ast.Constant)
                        and float(exp.value) == 2.0):
                    return F32(base * base)
                raise ValueError(f"unsupported power in {expr!r}")
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add):
                return F32(a + b)
            if isinstance(node.op, ast.Sub):
                return F32(a - b)
            if isinstance(node.op, ast.Mult):
                return F32(a * b)
            if isinstance(node.op, ast.Div):
                return F32(a / b)
        raise ValueError(f"unsupported node {ast.dump(node)} in {expr!r}")

    # Python reads the literal text exactly: re-read every Constant from the
    # source segment so 0.070 is the decimal 0.070, not a binary64.
    def literal_from_source(node):
        return ast.get_source_segment(expr, node)

    class _Exact(ast.NodeTransformer):
        def visit_Constant(self, node):
            return ast.copy_location(
                ast.Constant(value=literal_from_source(node)), node)

    tree = _Exact().visit(tree)
    return ev(tree)


def derived_words() -> dict[str, int]:
    """Every row's word, re-derived from its expression."""
    names: dict[str, np.float32] = {}
    words = {}
    for name, expr, _ in TEMF_CONSTANTS:
        value = fold(expr, names)
        names[name] = value
        words[name] = int(F32(value).view(np.uint32))
    return words


def cuda_table() -> str:
    """The ``TEMFC`` block of woof/core/kernels/temf.cu, byte for byte."""
    lines = [
        "// BEGIN TEMF CONSTANTS -- written by gpuwm/core/temf_constants.py"
        " (cuda_table); do not edit",
        f"#define TEMF_NCONST {len(TEMF_CONSTANTS)}",
        "__constant__ unsigned int TEMFC[TEMF_NCONST] = {",
    ]
    for index, (name, expr, word) in enumerate(TEMF_CONSTANTS):
        lines.append(f"    0x{word:08X}u,  // [{index:2d}] {name} = {expr}")
    lines.append("};")
    for index, (name, _, _) in enumerate(TEMF_CONSTANTS):
        lines.append(f"#define TC_{name} __uint_as_float(TEMFC[{index}])")
    lines.append("// END TEMF CONSTANTS")
    return "\n".join(lines) + "\n"


def table_sha256() -> str:
    return hashlib.sha256(cuda_table().encode("ascii")).hexdigest()


#: The pinned table.  tests/test_temf_constants.py holds it equal to
#: :func:`table_sha256`, so editing a row without re-pinning fails there.
TEMF_CONSTANTS_SHA256 = (
    "d8a710b986e517a39197c7af9f13009ba30869dfc65c053fc5ddb217eca4816d")


def fortran_probe() -> str:
    """A gfortran program printing every row's word, for the oracle check."""
    out = ["program temf_constants_probe", "  implicit none"]
    for name, expr, _ in TEMF_CONSTANTS:
        out.append(f"  real, parameter :: {name} = {expr}")
    for name, _, _ in TEMF_CONSTANTS:
        out.append(f"  write(*, '(a, 1x, z8.8)') '{name}', transfer({name}, 0)")
    out.append("end program temf_constants_probe")
    return "\n".join(out) + "\n"


def splice_cuda(text: str) -> str:
    """``text`` (temf.cu) with its constants block replaced by
    :func:`cuda_table`."""
    begin = text.index("// BEGIN TEMF CONSTANTS")
    marker = "// END TEMF CONSTANTS\n"
    end = text.index(marker) + len(marker)
    return text[:begin] + cuda_table() + text[end:]


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "--write-cuda":
        from pathlib import Path
        path = Path(argv[2])
        path.write_text(splice_cuda(path.read_text(encoding="utf-8")),
                        encoding="utf-8", newline="\n")
        return 0
    if argv[1:] == ["--cuda"]:
        sys.stdout.write(cuda_table())
    elif argv[1:] == ["--fortran-probe"]:
        sys.stdout.write(fortran_probe())
    elif argv[1:] == ["--sha256"]:
        print(table_sha256())
    elif argv[1:] == ["--derive"]:
        for name, word in derived_words().items():
            print(f"{name} {word:08X}")
    elif argv[1:] == ["--table"]:
        for name, _, word in TEMF_CONSTANTS:
            print(f"{name} {word:08X}")
    else:
        print("usage: python -m woof.core.temf_constants "
              "--cuda|--fortran-probe|--sha256|--derive", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
