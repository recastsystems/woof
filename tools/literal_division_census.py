"""A146: every float division by a compile-time constant the compiler sees.

NVRTC's NVVM, targeting compute_100 and compute_120 (Blackwell) under
``-ftz=true`` (which CuPy appends to every ``RawModule``), replaces a float
division by a compile-time constant with a multiplication by the rounded
reciprocal: ``x / 3.0f`` becomes ``mul.ftz.f32 x, 0f3EAAAAAB``, which is not
IEEE division (a third of all ``x / 3.0f`` quotients land one ULP off).
Every NVRTC measured (12.9, 13.0, 13.3, 13.4) does it; compute_90 and older
keep ``div.rn``.  A source grep cannot find every such site: a divisor can
reach the division through a macro, a ``const``, an integer literal, a
constant-folded product, or a helper inlined with a literal argument.  The
compiler can.  This census compiles each production translation unit for
compute_89, where the same division survives as ``div.rn[.ftz].f32`` with an
immediate divisor, and lists those instructions with the source line
``-lineinfo`` attaches to them.  A power-of-two, zero or infinite divisor
has an exact reciprocal, so either spelling gives the IEEE quotient, and it
is not listed.

The chosen form, ``__fdiv_rn(x, C)``, reaches PTX as a ``div.rn`` whose
divisor is a register, so it is never listed.

A compute_89 immediate is not the whole set compute_120 rewrites.  A divisor
compute_89 keeps in a register can still be a constant on compute_120: NVRTC
13.4.92 constant-folds ``tgammaf(4.0f)`` there, which compute_89 calls at run
time, and a pair of constant divisions in the two arms of a branch can reach
compute_89 as one ``div.rn`` of a selected register.  :func:`rewrite_sites`
compiles the unit for compute_120 twice, under ``-ftz=false`` (where a float
division by a constant stays a ``div.rn``) and under ``-ftz=true``, and lists
every source line whose IEEE quotients (``div.rn`` and ``rcp.rn``) fall under
``-ftz=true`` while its multiplies (or fused multiply-adds) gain an immediate
with an inexact reciprocal there: the division became a reciprocal multiply.
Constant operands are counted, including one-write ``mov.f32`` registers
within each PTX function. An existing coefficient of the same value must
not hide another occurrence introduced by a reciprocal rewrite.

The reference is the same architecture (A174).  It used to be compute_89,
and NVRTC 12.9.86, the default ``recast-woof[gpu]`` compiler, compiles ``powf`` and
``logf`` differently for the two architectures (as separate functions on
compute_120, inlined on compute_89), so lines of noah, nssl2_nucond, shinhong
and thompson that hold no constant division lost ``div.rn`` and gained
immediates between the two and were listed.  Over every production unit under
both compilers, the flush-mode comparison listed one line that is not a
rewrite, a reciprocal: NVRTC 13.4.92 compiles thompson.cu's
``1.0f / (1004.0f * (1.0f + 0.887f * qv0))`` as a ``div.rn`` under
``-ftz=false`` and as an ``rcp.rn`` under ``-ftz=true``, its sign folded into
a new multiply by -1004.  ``rcp.rn`` is the IEEE reciprocal, so it is counted
as the quotient it is, and that line is not listed.  The flush modes can
still inline differently (13.4.92 compiles two of p3_device's functions out
of line under ``-ftz=true`` only; no line of it is listed), so a new listing
is read from the PTX before it is taken for a rewrite.

Every source is read from one tree, named by ``root`` (A193): the gate tests
pass the repository root they run from, and ``--root`` defaults to the tree
holding this tool.  The loader's own composition functions assemble each
unit from that tree's files, so a woof imported from somewhere else (an
installed wheel beside a source tests tree, the B200 bench shape) supplies
no source.

Usage (needs CuPy's NVRTC; no GPU is touched when CUDA is hidden)::

    CUDA_VISIBLE_DEVICES= python -m tools.literal_division_census [--root tree]
        [--json out]
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import struct
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

#: The tree holding this tool: the command line's default ``--root`` only.
#: Every function below reads the tree it is given.
ROOT = Path(__file__).resolve().parents[1]


def kernel_dir(root) -> Path:
    """The kernel source directory of the tree at ``root``."""
    return Path(root) / "woof" / "core" / "kernels"


#: Units compiled with ``--ftz=false`` through ``compile_using_nvrtc``: the
#: rewrite needs ``-ftz=true``, so their constant divisions stay IEEE.
FTZ_FALSE_UNITS = ("rrtmg_legacy", "rrtmg_legacy_device", "rrtmg_lw",
                   "rrtmg_mcica", "rrtmg_sw")

_DIV_IMM = re.compile(
    r"^\s*div\.rn(?:\.ftz)?\.f32\s+%\w+,\s*%\w+,\s*0[fF]([0-9A-Fa-f]{8});")
_DIV_ANY = re.compile(r"^\s*div\.rn(?:\.ftz)?\.f32\s")
#: An IEEE-rounded float quotient: a ``div.rn``, or the ``rcp.rn`` a
#: reciprocal of a runtime value compiles to under one flush mode and not
#: the other (A174).
_QUOTIENT = re.compile(r"^\s*(?:div|rcp)\.rn(?:\.ftz)?\.f32\s")
_MUL_OP = re.compile(r"^\s*(?:mul|fma)(?:\.rn)?(?:\.ftz)?\.f32\s")
_IMMEDIATE = re.compile(r"(?<![\w.])0[fF]([0-9A-Fa-f]{8})\b")
_LOC = re.compile(r"^\s*\.loc\s+(\d+)\s+(\d+)\s+(\d+)(.*)$")
_INLINED_AT = re.compile(r"inlined_at\s+(\d+)\s+(\d+)\s+(\d+)")

_FUNCTION_START = re.compile(
    r"(?=^[ \t]*(?:(?:\.visible|\.weak|\.extern)\s+)?\.(?:entry|func)\b)",
    re.MULTILINE)
_INSTRUCTION = re.compile(
    r"^\s*(?:@\S+\s+)?([A-Za-z][\w.]*)\s+([^;]+);")
_DESTINATION = re.compile(r"^\s*([({][^)}]*[)}]|[^,]+)")
_REGISTER = re.compile(r"%\w+")


@dataclass
class Unit:
    key: str
    source: str
    options: tuple[str, ...]
    #: (first assembled line, file label, first line in that file)
    segments: list[tuple[int, str, int]] = field(default_factory=list)

    def locate(self, line: int) -> tuple[str, int]:
        label, base, first = "<assembled>", 1, 1
        for start, lab, file_first in self.segments:
            if line >= start:
                label, base, first = lab, start, file_first
        return label, line - base + first


def _lines(text: str) -> int:
    return text.count("\n")


def _segmented(key, pieces, options):
    """Assemble ``pieces`` [(label, text)] and remember where each starts."""
    segs, line = [], 1
    for label, text in pieces:
        segs.append((line, label, 1))
        line += _lines(text)
    return Unit(key, "".join(text for _label, text in pieces), options, segs)


def _preamble_pieces(kdir: Path):
    from woof.core.constants import CUDA_DEFINES
    defines = "".join(f"#define {k} {float(v)!r}f\n"
                      for k, v in CUDA_DEFINES.items())
    common = (kdir / "common.cuh").read_text(encoding="utf-8")
    return [("<CUDA_DEFINES>", defines), ("woof/core/kernels/common.cuh",
                                          common + "\n")]


def production_units(root) -> list[Unit]:
    """Every runtime translation unit, with its caller arithmetic options,
    from the tree at ``root`` (its repository root).

    Every source text is read under ``root`` (A193).  The loader's own
    composition functions assemble each unit from that tree's kernel files,
    and each unit the census labels line by line is checked against them,
    so the census compiles what the loader would compile from this tree.
    """
    from woof.core import kernels as K
    from woof.core import noahmp_kernel_sources as N

    kdir = kernel_dir(Path(root).resolve())
    if not (kdir / "common.cuh").is_file():
        raise FileNotFoundError(f"{root} holds no woof kernel tree at {kdir}")
    units: list[Unit] = []
    pre = _preamble_pieces(kdir)
    assert "".join(t for _l, t in pre) == K._preamble(kdir), \
        "census preamble drifted from the loader's"

    def kfile(name):
        return (f"woof/core/kernels/{name}",
                (kdir / name).read_text(encoding="utf-8"))

    # Static macro aliases are compiler input owned by the loader itself.
    # Attribute their literal bytes to their actual source lines instead of
    # maintaining a second hand-copied alias list. Full assembly equality
    # below still catches an omitted or changed byte, in either mode.
    loader_path = Path(root).resolve() / "woof/core/kernels/__init__.py"
    loader_ast = ast.parse(loader_path.read_text(encoding="utf-8"))
    composer = next(n for n in loader_ast.body
                    if isinstance(n, ast.FunctionDef) and n.name == "_extra_header_text")
    loader_macros = []
    for node in ast.walk(composer):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            lines = node.value.splitlines()
            if (lines and any(line.strip() for line in lines)
                    and all(not line.strip() or line.startswith(("#define ", "#undef "))
                            for line in lines)):
                loader_macros.append((f"woof/core/kernels/__init__.py:{node.lineno}",
                                      node.value))

    def headers(name):
        # Resolve the active loader composition, including mode-selected
        # headers. Keep labels for PTX attribution and require exact bytes.
        remaining = K._extra_header_text(name, kdir)
        parts = []
        from woof.wrf_exact import ENABLED
        if not ENABLED and name == "nssl2":
            # Inline math aliases are loader text rather than a header file.
            return [("<nssl2 default math>", remaining)]
        while remaining:
            matches = [(p.name, p.read_text(encoding="utf-8"))
                       for p in sorted(p for p in kdir.iterdir() if p.is_file())]
            matches.extend(loader_macros)
            if "gfdleta_support.cuh" in K.EXTRA_HEADERS.get(name, ()):
                from woof.wrf_exact import referee_glibc
                matches.append(("<GPUWM_GFDLETA_GLIBC243>",
                                f"#define GPUWM_GFDLETA_GLIBC243 {int(referee_glibc() >= (2, 43))}\n"))
            if name == "gfs_sas":
                from woof.numerics import is_strict
                matches.append(("<SAS_STRICT>",
                                f"#define SAS_STRICT {int(is_strict())}\n"))
            if name in ("gsfcgce4ice", "gsfcgce4ice_aux"):
                from woof.wrf_exact import referee_glibc
                if referee_glibc() >= (2, 41):
                    matches.append(("<GPUWM_G4_CORE_LOG10>",
                                    "#define GPUWM_G4_CORE_LOG10 1\n"))
            if name == "cu_osas" and ENABLED:
                matches.append(("<GPUWM_COLUMN_R8_STRICT>",
                                "#define GPUWM_COLUMN_R8_STRICT 1\n"))
            if not ENABLED and name in ("nssl2_shape_nucond", "nssl2_shape_gs",
                                        "nssl2_shape_driver", "nssl2_shape_diagnostics"):
                # Loader text (K._extra_header_text): the fast NSSL shape
                # units alias the WRF libm names to CUDA's, as the fast nssl2
                # unit does; their __fdiv_rn spellings stay IEEE (A146).
                matches.append(("<nssl2 default math>",
                                "#define gfk_pow powf\n#define gfk_exp expf\n#define gfk_log logf\n"))
            if not ENABLED and name in ("wind_mav", "wind_legacy"):
                # Loader text: the fast wind units take CUDA's libm.
                matches.append(("<wind fast math>",
                                "#define GPUWM_WIND_FAST 1\n#define glibc_sinf sinf\n"
                                "#define glibc_cosf cosf\n#define glibc_asinf asinf\n"
                                "#define glibc_acosf acosf\n#define gfk_exp expf\n"
                                "#define plm_exp exp\n#define uw_acos acos\n"))
            if name == "sas" and K._sas_strict_numerics():
                from woof.wrf_exact import referee_correctly_rounded_trig
                if referee_correctly_rounded_trig():
                    matches.append(("<SAS correctly rounded trig>",
                                    "\n#undef wrf_atan_f32\n#define wrf_atan_f32 glibc_atanf\n"))
            if name == "temf_sfclay":
                from woof.core.temf_constants import cuda_table
                matches.append(("<TEMF constants>", cuda_table()))
            if name in K._TABLE_HEADERS:
                relative = K._TABLE_HEADERS[name][0]
                matches.append((f"woof/data/{relative}",
                                K._table_header_text(name, data_root=kdir.parents[1] / "data")))
            if name == "keps":
                from woof.table_assets import kernel_parameter_source_pieces
                matches.extend(kernel_parameter_source_pieces(
                    name, data_root=kdir.parents[1] / "data"))
            # Declared coefficient headers live beside the scanned
            # kernels under this same tree, not the imported package.
            for header in K.EXTRA_HEADERS.get(name, ()):
                if header.startswith("../../data/"):
                    path = (kdir / header).resolve()
                    matches.append((path.relative_to(Path(root).resolve()).as_posix(),
                                    path.read_text(encoding="utf-8")))
            if name == "sfs_output":
                matches = [("<SFS geometry from smag2d.cu>", K._sfs_smag_geometry_source(text))
                           if n == "smag2d.cu" else (n, text)
                           for n, text in matches]
            if name == "flg_column":
                if not ENABLED:
                    # Attribute the production column's exact fast/rounded
                    # boundary to its generated source, retaining every byte.
                    stream = "// Generated from pristine WRF FLG stream solver statements."
                    cloud = "// Generated from WRF module_ra_flg.F SHA256"
                    rounded = "\n#undef __fadd_rn\n#undef __fsub_rn\n#undef __fmul_rn\n#undef __fdiv_rn\n"
                    matches = [(n, (K._flg_fast_math_aliases() + text.replace(
                        stream, rounded + stream, 1).replace(
                        cloud, K._flg_fast_math_aliases() + cloud, 1))
                        if n == "flg_workspace_generated.cuh" else text)
                        for n, text in matches]
                    matches.append(("<FLG column rounded math>",
                                    K._flg_fast_math_aliases(rounded_operators=True)))
                matches = [(n, K._flg_column_noinline(text)) for n, text in matches]
            if not ENABLED and name in ("flg_column", "flg_driver"):
                matches.append(("<FLG default math>", K._flg_fast_math_aliases()))
            if not ENABLED and name == "smag2d":
                # smag2d's default composition is glibc_flt32.cuh with the
                # loader's own gfk_exp rewrite (K._smag2d_default_gfk_exp);
                # attribute it to that file, through that same function.
                matches = [(n, K._smag2d_default_gfk_exp(text) if n == "glibc_flt32.cuh" else text)
                           for n, text in matches]
            matches = [(n, text) for n, text in matches
                       if text and remaining.startswith(text)]
            if not matches:
                raise AssertionError(f"unattributed loader header for {name}")
            filename, text = max(matches, key=lambda item: len(item[1]))
            label = filename if "/" in filename else f"woof/core/kernels/{filename}"
            parts.append((label, text))
            remaining = remaining[len(text):]
        return parts

    def unit_piece(name):
        from woof.wrf_exact import ENABLED
        filename = name + ".cu"
        if not ENABLED and name in K._DEFAULT_SOURCE_MODULES:
            filename = name + "_default.cu"
        source = K._unit_text(name, kdir)
        prefix = "#define NOAH_WRF_SCALAR_RCT 1\n"
        if name == "noah" and source.startswith(prefix):
            return [("<NOAH_WRF_SCALAR_RCT>", prefix),
                    (f"woof/core/kernels/{filename}", source[len(prefix):])]
        return [(f"woof/core/kernels/{filename}", source)]

    noahmp_parts = set()
    # These files compile through their own observed NVRTC composition. A
    # bare generic-loader source omits the shared SFIRE physics and libm.
    from woof.core import sfire_atm, sfire_core, sfire_moisture, sfire_phys, sfire_wind, sfire_coupler, sfire_spotting, sfire_ideal, sfire_ideal_atmos, chem_sfire, urban_bem
    dedicated_units = {
        "sfire_atm": (sfire_atm.module_source, sfire_atm.MODULE_OPTIONS),
        "sfire_core": (sfire_core.source, sfire_core.NVRTC_OPTIONS),
        "sfire_moisture": (sfire_moisture.module_source, sfire_moisture.MODULE_OPTIONS),
        "sfire_phys": (sfire_phys.module_source, sfire_phys.MODULE_OPTIONS),
        "sfire_wind": (sfire_wind.module_source, sfire_wind.MODULE_OPTIONS),
        "sfire_coupling": (sfire_coupler.module_source, sfire_coupler.MODULE_OPTIONS),
        "sfire_spotting": (sfire_spotting.module_source, sfire_spotting.MODULE_OPTIONS),
        "sfire_ideal": (sfire_ideal.module_source, sfire_ideal.MODULE_OPTIONS),
        "sfire_ideal_atmos": (sfire_ideal_atmos.module_source, sfire_ideal_atmos.MODULE_OPTIONS),
        "chem_sfire": (chem_sfire.module_source, chem_sfire.MODULE_OPTIONS),
        "urban_bep_bem": (urban_bem.module_source, urban_bem.MODULE_OPTIONS),
        # KF compiles on its own direct-NVRTC route (woof/core/kf.py
        # _kf_module) with KF_MODULE_OPTIONS, not the generic loader's.
        "kf": (lambda kernel_dir: K.module_source("kf", kernel_dir=kernel_dir),
               _kf_module_options()),
    }
    for parts in N.NOAHMP_TRANSLATION_UNITS.values():
        noahmp_parts.update(parts)
    for path in sorted(kdir.glob("*.cu")):
        name = path.stem
        if name.endswith("_default") and name[:-len("_default")] in K._DEFAULT_SOURCE_MODULES:
            continue  # Selected above under their runtime module names.
        if name in dedicated_units:
            source_fn, options = dedicated_units[name]
            units.append(Unit(f"kernels:{name}", source_fn(kernel_dir=kdir), tuple(options), []))
            continue
        if name in noahmp_parts:
            continue
        extra = headers(name)
        from woof.cuda_subnormals import MARKER
        prefix = [("<GPUWM_RETAIN_SUBNORMALS>", MARKER)] if name in K._URBAN_NO_FTZ_MODULES else []
        # module_source's own prefixes, in its order (K.module_source).
        if name == "pxsfclay":
            from woof.core.pxsfclay import source_constant_prefix
            prefix.append(("woof/data/pxsfclay/constants.json",
                           source_constant_prefix(kdir.parents[1] / "data" / "pxsfclay")))
        if name in ("deng", "mskf", "mskf_aerosol_replay"):
            from woof.numerics import is_strict
            if not is_strict():
                define = "GPUWM_DENG_FAST" if name == "deng" else "GPUWM_MSKF_FAST"
                prefix.append(("<" + define + ">", "#define " + define + " 1\n"))
        unit = _segmented(f"kernels:{name}", prefix + pre + extra + unit_piece(name),
                          K.module_options(name))
        if unit.source != K.module_source(name, kernel_dir=kdir):
            raise AssertionError(f"census source for {name} drifted")
        units.append(unit)
    # gf above 40 levels compiles with an integer tier; 60 is one such tier.
    extra = headers("gf")
    gf = _segmented("kernels:gf[GF_KMAX=60]",
                    pre + extra + [("<GF_KMAX>", "#define GF_KMAX 60\n"),
                                   kfile("gf.cu")], ("-std=c++17",))
    assert gf.source == K.module_source_int_defines(
        "gf", (("GF_KMAX", 60),), kernel_dir=kdir)
    units.append(gf)
    from woof.core.spp_kernel_sources import specialized_source
    for name, capacity in (("gf", 40), ("gf", 60), ("mynn_pbl", 40), ("mynn_surface", 40)):
        units.append(Unit(f"spp:{name}[capacity={capacity}]",
                          specialized_source(name, capacity=capacity, kernel_dir=kdir),
                          ("-std=c++17",), []))
    for name, parts in N.NOAHMP_TRANSLATION_UNITS.items():
        ru = N.runtime_unit(name, kernel_dir=kdir)
        pieces = ([] if ru.preamble_sha256 != N._sha(K._preamble(kdir))
                  else pre)
        pieces = pieces + [kfile(p + ".cu") for p in parts]
        unit = _segmented(f"noahmp:{name}", pieces, tuple(ru.options))
        if unit.source != ru.source:
            # Composition differs from the simple concatenation (a unit with
            # its own prefix); keep the real source and locate lines by
            # searching rather than guessing.
            unit = Unit(f"noahmp:{name}", ru.source, tuple(ru.options), [])
        units.append(unit)
    from woof.core import ruc_tier as RT
    for nzs in (6, 9):
        units.append(Unit(f"ruc_tier[nzs={nzs}]",
                          RT.ruc_fused_source(nzs, kernel_dir=kdir),
                          ("-std=c++17",), []))
    from woof.core import p3_device as P3
    # By file name in this tree: the imported package's paths name another
    # tree whenever woof is installed beside the one scanned (A193).
    p3 = _segmented("p3_device", pre + [
        kfile(P3._LIBM_SOURCE.name), kfile(P3._P3_SOURCE.name)],
        tuple(P3.DEFAULT_OPTIONS))
    assert p3.source == P3.p3_source(kernel_dir=kdir)
    units.append(p3)
    return units


def compile_ptx(source: str, options: tuple[str, ...]) -> str:
    # Keyed by its options (A160): NVRTC's cache ignores -ftz for a fixed
    # program name, so one planted source compiled under both flush modes
    # on a host with a GPU could read the other mode's PTX.
    from woof.nvrtc_cache_key import compile_program
    ptx = compile_program(source, options, target="ptx")
    return ptx.decode() if isinstance(ptx, bytes) else ptx


def _is_exact_divisor(bits: int) -> bool:
    """A power of two, zero or infinity: ``x * (1/C)`` equals ``x / C``.

    Division by a power of two only moves the exponent (the reciprocal is
    exact), and a zero or infinite divisor has an exact reciprocal too, so
    the rewrite gives the IEEE quotient for these (a zero divisor with a zero
    numerator is NaN either way)."""
    return (bits & 0x007FFFFF) == 0


def _kf_module_options() -> tuple[str, ...]:
    from woof.core.kf import KF_MODULE_OPTIONS
    return tuple(KF_MODULE_OPTIONS)


def unflushed_route(options) -> bool:
    """Does a unit compile with ``--ftz=false`` on its production route?

    A146's rewrite (``x / C`` into ``x * RN(1/C)``) needs ``-ftz=true``.  A
    unit whose own options carry ``--ftz=false`` (KF, Grell 3D, New SAS,
    lightning, the surface layers and the other unflushed routes in
    woof.core.kernels) keeps ``div.rn`` for a constant divisor, which is the
    IEEE quotient, so both census passes report no site for it rather than
    compiling it under a flush mode production never uses.  The source
    scan's FTZ_FALSE_SOURCES is the same rule at the source layer.
    """
    return any(str(option).replace("--", "-") == "-ftz=false"
               for option in options)


def census_unit(unit: Unit, arch: str = "compute_89") -> dict:
    """Constant-divisor float divisions of one unit, with source lines."""
    if unflushed_route(unit.options):
        return {"unit": unit.key, "arch": arch, "div_rn_f32": None,
                "constant_divisor_sites": [], "unflushed_route": True}
    from woof.cuda_subnormals import effective_options
    opts = effective_options(unit.source, tuple(unit.options) +
                             ("-ftz=true", "-lineinfo", f"-arch={arch}"))
    ptx = compile_ptx(unit.source, opts)
    sites, loc, total = [], None, 0
    for text in ptx.splitlines():
        m = _LOC.match(text)
        if m:
            loc = (int(m.group(2)), int(m.group(3)),
                   _INLINED_AT.findall(m.group(4)))
            continue
        if _DIV_ANY.match(text):
            total += 1
        m = _DIV_IMM.match(text)
        if not m:
            continue
        bits = int(m.group(1), 16)
        if _is_exact_divisor(bits):
            continue
        divisor = struct.unpack("<f", struct.pack("<I", bits))[0]
        line, column, chain = loc if loc else (0, 0, [])
        label, file_line = unit.locate(line)
        outer = None
        if chain:
            _f, oline, _c = chain[-1]
            outer = unit.locate(int(oline))
        sites.append({"file": label, "line": file_line, "column": column,
                      "assembled_line": line, "divisor": repr(divisor),
                      "inlined_into": list(outer) if outer else None})
    return {"unit": unit.key, "arch": arch, "div_rn_f32": total,
            "constant_divisor_sites": sites}


def _constant_registers(ptx: str) -> dict[str, int]:
    """Registers written once, by ``mov.f32`` of a float immediate.

    The caller passes one PTX function, since register names are local to
    it.  Reject every multiply-written register, including conditional
    assignments, rather than guessing a constant across control flow.
    """
    writes: dict[str, list[int | None]] = {}
    for text in ptx.splitlines():
        instruction = _INSTRUCTION.match(text)
        if not instruction:
            continue
        opcode, operands = instruction.groups()
        destination = _DESTINATION.match(operands)
        if not destination:
            continue
        registers = _REGISTER.findall(destination.group(1))
        value = None
        if opcode == "mov.f32" and len(registers) == 1:
            immediate = _IMMEDIATE.fullmatch(
                operands[destination.end():].lstrip(" ,"))
            if immediate:
                value = int(immediate.group(1), 16)
        for register in registers:
            writes.setdefault(register, []).append(value)
    return {register: values[0] for register, values in writes.items()
            if len(values) == 1 and values[0] is not None}


def _line_counts(ptx: str) -> dict[tuple[int, int], tuple[int, Counter[int]]]:
    """(file, line) -> (IEEE quotient count, constant float operands of mul
    and fma.f32), by the innermost source line ``-lineinfo`` gives each
    instruction.  A one-write ``mov.f32`` register and a direct immediate
    are the same operand, as NVRTC's targets can use either spelling.
    Count occurrences so a new reciprocal cannot hide behind an existing
    coefficient with the same value."""
    counts: dict[tuple[int, int], list] = {}
    for function in _FUNCTION_START.split(ptx):
        constants = _constant_registers(function)
        loc = None
        for text in function.splitlines():
            m = _LOC.match(text)
            if m:
                loc = (int(m.group(1)), int(m.group(2)))
                continue
            if loc is None:
                continue
            if _QUOTIENT.match(text):
                counts.setdefault(loc, [0, Counter()])[0] += 1
            elif _MUL_OP.match(text):
                operands = text.split(",", 1)[1]
                values = [int(bits, 16)
                          for bits in _IMMEDIATE.findall(operands)]
                values.extend(constants[register]
                              for register in _REGISTER.findall(operands)
                              if register in constants)
                counts.setdefault(loc, [0, Counter()])[1].update(values)
    return {key: (n, imms) for key, (n, imms) in counts.items()}


def _plain_division_on(line_text: str) -> bool:
    """Does this source line hold a ``/`` outside comments and strings?  A
    line whose only divisions are ``__fdiv_rn`` calls (or helpers built on
    it) cannot be rewritten; compute_120 folding such a call on constants
    gives the IEEE quotient."""
    code = line_text
    for pattern in (r"//.*$", r"/\*.*?\*/", r'"(?:\\.|[^"\\])*"'):
        code = re.sub(pattern, " ", code)
    return "/" in code


def _flush_counts(unit: Unit, arch: str, flush: str):
    """The per-line quotient and immediate counts of ``unit`` compiled
    for ``arch`` under one flush mode (``-ftz=false`` or ``-ftz=true``)."""
    base = tuple(unit.options) + ("-lineinfo", f"-arch={arch}")
    return _line_counts(compile_ptx(unit.source, base + (flush,)))


def rewrite_sites(unit: Unit, arch: str = "compute_120", counts=None) -> dict:
    """Source lines whose float division ``arch`` compiles under
    ``-ftz=true`` as a multiply by an immediate where the same architecture
    under ``-ftz=false`` keeps a ``div.rn``: the line loses an IEEE quotient
    (``div.rn`` or ``rcp.rn``) under ``-ftz=true`` and its multiplies gain an
    immediate the reference does not use there, one whose reciprocal is not
    exact (a power of two, zero or infinity rewrite to the IEEE quotient).
    A line that loses the division by folding a constant ``__fdiv_rn`` is
    not listed.

    The reference is the same architecture so that only the flush mode
    differs between the two builds (A174): a compute_89 reference read
    NVRTC 12.9.86's compute_120 code for ``powf`` and ``logf``, which is not
    its compute_89 code, as rewritten divisions."""
    if unflushed_route(unit.options):
        return {"unit": unit.key, "arch": arch, "rewritten_sites": [],
                "unflushed_route": True}
    if counts is None:
        before = _flush_counts(unit, arch, "-ftz=false")
        after = _flush_counts(unit, arch, "-ftz=true")
    else:
        # The two compiles were run elsewhere (census_tree's workers):
        # (-ftz=false counts, -ftz=true counts) of this unit under arch.
        before, after = counts
    source_lines = unit.source.splitlines()
    sites = []
    for key in sorted(set(before) | set(after)):
        div_old, imm_old = before.get(key, (0, Counter()))
        div_new, imm_new = after.get(key, (0, Counter()))
        gained = sorted(bits for bits in imm_new - imm_old
                        if not _is_exact_divisor(bits))
        if div_new >= div_old or not gained:
            continue
        if not 0 < key[1] <= len(source_lines):
            continue    # no source line to attribute a plain division to
        text = source_lines[key[1] - 1]
        if not _plain_division_on(text):
            continue
        label, file_line = unit.locate(key[1])
        sites.append({"file": label, "line": file_line,
                      "assembled_line": key[1],
                      "quotients": [div_old, div_new],
                      "gained_immediates": [
                          repr(struct.unpack("<f", struct.pack("<I", b))[0])
                          for b in gained]})
    return {"unit": unit.key, "arch": arch,
            "flush_modes": ["-ftz=false", "-ftz=true"],
            "rewritten_sites": sites}


def census(root, units=None) -> list[dict]:
    return [census_unit(u) for u in (units or production_units(root))]


#: The two whole-tree passes :func:`census_tree` can run, by name.
_PASSES = ("immediate", "rewrites")

#: Per worker process: the units of the tree it was started for, so each
#: unit's source crosses the process boundary once (as an index), not once
#: per call.
_WORKER_UNITS: dict[str, list[Unit]] = {}


def _worker_init(root: str) -> None:
    # The worker is a fresh interpreter (spawn): put the tree under test on
    # its path so ``woof`` and ``tools`` resolve to that tree, assemble its
    # units once, and initialise NVRTC once here rather than on the first
    # compile of every unit.
    if root not in sys.path:
        sys.path.insert(0, root)
    from cupy.cuda import nvrtc
    nvrtc.getVersion()
    _WORKER_UNITS[root] = production_units(root)


def _worker_compile(job: tuple[str, int, str, str | None]):
    """One NVRTC compile: the immediate census of a unit (``flush`` None)
    or the line counts of a unit under one flush mode of the rewrite pass.
    ``None`` for a unit NVRTC refuses alone: a fragment compiled through
    another unit (p3 via p3_device) or an --ftz=false RRTMG source; the
    census covers the fragments through their real units and the callers
    count only the units that compiled."""
    root, index, arch, flush = job
    units = _WORKER_UNITS.get(root)
    if units is None:
        _worker_init(root)
        units = _WORKER_UNITS[root]
    try:
        if flush is None:
            return census_unit(units[index], arch)
        return _flush_counts(units[index], arch, flush)
    except Exception:
        return None


def census_tree(root, pass_name: str = "immediate", arch: str | None = None,
                workers: int | None = None) -> list[tuple[str, dict | None]]:
    """Run one census pass over every production unit of the tree at
    ``root``, compiling concurrently, one NVRTC per worker.

    Returns ``[(unit.key, row)]`` in :func:`production_units` order, with
    ``None`` for a unit NVRTC refused alone, so the result is the serial
    loop's (``census_unit`` or ``rewrite_sites`` per unit), whatever the
    number of workers.  The pass is ``"immediate"`` (:func:`census_unit`,
    compute_89) or ``"rewrites"`` (:func:`rewrite_sites`, compute_120); the
    flags each unit compiles with are the same as in the serial loop, since
    the same functions compile it.  The rewrite pass's two compiles of a
    unit (one per flush mode) are separate tasks, so the longest unit
    bounds the wall by one compile, not two; the comparison runs here.

    Each worker is a fresh interpreter (the spawn start method): NVRTC
    holds process-wide state that a forked child inherits half-built, and
    CuPy's NVRTC binding in a forked worker can block on it.  ``workers``
    defaults to one per core, capped at the number of tasks and at 32 (one
    compile takes one core; more lanes than cores only add memory).  The
    largest sources are submitted first, so the slowest compiles start at
    once instead of after the queue has drained around them.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    if pass_name not in _PASSES:
        raise ValueError(f"pass must be one of {_PASSES}, not {pass_name!r}")
    arch = arch or ("compute_120" if pass_name == "rewrites"
                    else "compute_89")
    root = str(Path(root).resolve())
    units = production_units(root)
    rows: list = [None] * len(units)
    jobs: list[tuple[str, int, str, str | None]] = []
    for index, unit in enumerate(units):
        if pass_name == "immediate":
            if unflushed_route(unit.options):
                rows[index] = census_unit(unit, arch)   # no compile
            else:
                jobs.append((root, index, arch, None))
        elif unflushed_route(unit.options):
            rows[index] = rewrite_sites(unit, arch)      # no compile
        else:
            jobs.append((root, index, arch, "-ftz=false"))
            jobs.append((root, index, arch, "-ftz=true"))
    jobs.sort(key=lambda job: -len(units[job[1]].source))
    if workers is None:
        cores = getattr(os, "process_cpu_count", os.cpu_count)() or 1
        workers = max(1, min(len(jobs), cores, 32))
    results = {}
    if jobs:
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=get_context("spawn"),
                                 initializer=_worker_init,
                                 initargs=(root,)) as pool:
            for job, result in zip(jobs, pool.map(_worker_compile, jobs)):
                results[job[1:]] = result
    for index, unit in enumerate(units):
        if pass_name == "immediate":
            if (index, arch, None) in results:
                rows[index] = results[(index, arch, None)]
        elif (index, arch, "-ftz=false") in results:
            before = results[(index, arch, "-ftz=false")]
            after = results[(index, arch, "-ftz=true")]
            if before is not None and after is not None:
                rows[index] = rewrite_sites(unit, arch, counts=(before, after))
    return [(unit.key, row) for unit, row in zip(units, rows)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", help="write the full census here")
    ap.add_argument("--root", default=str(ROOT),
                    help="repository root of the tree to census (default: "
                         "the tree holding this tool)")
    ap.add_argument("--arch", default=None,
                    help="compute_89 for the immediate census, compute_120 "
                         "for --rewrites (the defaults)")
    ap.add_argument("--rewrites", action="store_true",
                    help="compare -ftz=false with -ftz=true line by line "
                         "on --arch instead")
    args = ap.parse_args(argv)
    rows, failed = [], []
    key = "rewritten_sites" if args.rewrites else "constant_divisor_sites"
    arch = args.arch or ("compute_120" if args.rewrites else "compute_89")
    for unit in production_units(args.root):
        try:
            rows.append(rewrite_sites(unit, arch) if args.rewrites
                        else census_unit(unit, arch))
        except Exception as exc:  # a unit NVRTC refuses is reported, not hidden
            failed.append((unit.key, str(exc).splitlines()[0][:200]))
    for unit_key, why in failed:
        print(f"COMPILE FAILED {unit_key}: {why}")
    total = 0
    for row in rows:
        n = len(row[key])
        total += n
        if n:
            print(f"{row['unit']}: {n}")
            for site in row[key]:
                print(f"    {site['file']}:{site['line']}")
    print(f"total {key.replace('_', ' ')}: {total}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1) + "\n",
                                   encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
